"""
Correlation driver: compute (X_all, X_key) on gov_report calibration and
Y_jsd on RULER answer tokens, for N candidate quant configs of
Llama-3.1-8B-Instruct.

Outputs:
  <save>/results.csv       per-config aggregated metrics
  <save>/per_batch.json    raw per-(config,batch/instance) metrics
  <save>/dense_keytokens/  optional dump of key-token indices

This script does ONE outer pass over data: for each calib batch / RULER
instance, dense is forwarded once and held on-device, then every candidate
config is forwarded and JSD is accumulated against the dense logits. Avoids
caching dense logits to disk.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

# Make sibling modules importable when launched from search/ root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluator import LlamaEvaluator
from utils.data import get_loader
from utils.eval import get_tokenizer
from utils.func import get_hfmodel, clean_up, init_accelerator
from utils.loss import find_key_token, cal_overlap, compute_offsets
from utils.ruler_utils import niah_utils, vt_utils, cwe_utils, fwe_utils, qa_utils


RULER_TASK_FN = {
    "niah_single_1": niah_utils.niah_single_1,
    "niah_single_2": niah_utils.niah_single_2,
    "niah_single_3": niah_utils.niah_single_3,
    "niah_multikey_1": niah_utils.niah_multikey_1,
    "niah_multikey_2": niah_utils.niah_multikey_2,
    "niah_multikey_3": niah_utils.niah_multikey_3,
    "niah_multivalue": niah_utils.niah_multivalue,
    "niah_multiquery": niah_utils.niah_multiquery,
    "ruler_vt": vt_utils.get_vt_dataset,
    "ruler_cwe": cwe_utils.get_cw_dataset,
    "ruler_fwe": fwe_utils.fwe_download,
    "ruler_qa_squad": qa_utils.get_squad,
    "ruler_qa_hotpot": qa_utils.get_hotpotqa,
}


@torch.no_grad()
def jsd_pair(p_logits: torch.Tensor, q_logits: torch.Tensor, eps: float = 1e-7, chunk: int = 1024) -> torch.Tensor:
    """Token-wise JSD between two logit tensors of shape [N, V]. Returns [N]
    on the device of p_logits. Chunks along N to keep peak memory bounded.
    """
    if q_logits.device != p_logits.device:
        q_logits = q_logits.to(p_logits.device)
    N = p_logits.shape[0]
    out = torch.empty(N, dtype=torch.float32, device=p_logits.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        p = p_logits[s:e].float().softmax(-1)
        q = q_logits[s:e].float().softmax(-1)
        m = (0.5 * (p + q)).clamp_min(eps)
        log_m = m.log()
        log_p = p_logits[s:e].float().log_softmax(-1)
        log_q = q_logits[s:e].float().log_softmax(-1)
        kl_pm = (p * (log_p - log_m)).sum(-1)
        kl_qm = (q * (log_q - log_m)).sum(-1)
        out[s:e] = 0.5 * (kl_pm + kl_qm)
    return out


@torch.no_grad()
def quant_forward_strided(model, input_ids, stride: int):
    """Forward pass with use_cache=True and strided past_kv chunks. Required to
    exercise KIVI / per-step KV quantization during prefill on a long sequence.
    Returns full logits tensor [B, T, V].
    """
    if stride is None or stride <= 0:
        return model(input_ids).logits
    total_len = input_ids.shape[1]
    chunked = []
    past_kv = None
    for start in range(0, total_len, stride):
        end = min(start + stride, total_len)
        chunk_in = input_ids[:, start:end]
        out = model(chunk_in, past_key_values=past_kv, use_cache=True)
        chunked.append(out.logits)
        past_kv = out.past_key_values
    return torch.cat(chunked, dim=1)


def find_answer_token_positions(tokenizer, full_text: str, answer_text: str):
    """Return shift_logits indices (i.e., positions in input that *predict* answer tokens).
    Returns None if answer not localizable.
    """
    # Locate answer string in the combined text by char offset.
    pos = full_text.rfind(answer_text)
    if pos < 0:
        return None
    char_start = pos
    char_end = pos + len(answer_text)
    try:
        enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
    except (NotImplementedError, ValueError):
        return None
    input_ids = enc["input_ids"][0]
    # Token indices whose char span overlaps [char_start, char_end).
    answer_token_idx = []
    for i, (s, e) in enumerate(offsets):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        answer_token_idx.append(i)
    if not answer_token_idx:
        return None
    # In shift form (loss/JSD on next-token prediction), position i in shift_logits
    # predicts input_ids[i+1]. So the position predicting answer token at idx t
    # is shift index t-1. Drop t==0 (no preceding context to predict it).
    shift_idx = [t - 1 for t in answer_token_idx if t >= 1]
    if not shift_idx:
        return None
    return input_ids, torch.tensor(shift_idx, dtype=torch.long)


def build_ruler_instances(tokenizer, model_id, tasks, n_per_task, length, seed):
    """Return list of dicts: {task, input_ids, answer_shift_idx}."""
    instances = []
    for task in tasks:
        ds = RULER_TASK_FN[task](model=model_id, max_seq_lengths=[length], num_samples=max(n_per_task * 2, n_per_task))["test"]
        ds = ds.shuffle(seed)
        kept = 0
        for i in range(len(ds)):
            if kept >= n_per_task:
                break
            doc = ds[i]
            ans_list = doc["outputs"]
            if isinstance(ans_list, str):
                ans_list = [ans_list]
            answer = ans_list[0]
            # Compose the teacher-forced sequence: input + " " + gen_prefix + " " + answer
            full = doc["input"] + " " + doc["gen_prefix"] + " " + answer
            res = find_answer_token_positions(tokenizer, full, answer)
            if res is None:
                continue
            input_ids, shift_idx = res
            instances.append({
                "task": task,
                "doc_idx": i,
                "input_ids": input_ids.unsqueeze(0),  # [1, T]
                "answer_shift_idx": shift_idx,
                "answer_text": answer,
            })
            kept += 1
        if kept < n_per_task:
            print(f"[warn] {task}: only {kept}/{n_per_task} instances usable", flush=True)
    return instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=str, default="0,1,2,3")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--quant_model_paths", type=str, nargs="+", required=True)
    ap.add_argument("--w_bits", type=int, nargs="+", required=True)
    ap.add_argument("--w_method", type=str, default="hqq")
    ap.add_argument("--kv_method", type=str, default="kivi")
    ap.add_argument("--w_group_size", type=int, default=128)
    ap.add_argument("--k_group_size", type=int, default=128)
    ap.add_argument("--v_group_size", type=int, default=128)
    ap.add_argument("--k_quant_scheme", type=str, default="channel")
    ap.add_argument("--v_quant_scheme", type=str, default="token")
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="float16")
    # calibration
    ap.add_argument("--calib_dataset", type=str, default="gov_report")
    ap.add_argument("--n_sample", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--min_seqlen", type=int, default=8192)
    ap.add_argument("--data_batch_size", type=int, default=1)
    # key tokens
    ap.add_argument("--trunc_len", type=int, default=512)
    ap.add_argument("--sliding_window", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=-2.0)
    # ruler
    ap.add_argument("--ruler_tasks", type=str, nargs="+", default=["niah_single_1", "ruler_vt"])
    ap.add_argument("--ruler_n_per_task", type=int, default=30)
    ap.add_argument("--ruler_length", type=int, default=8192)
    # quant forward: strided with past_kv to exercise KV quantization
    ap.add_argument("--stride", type=int, default=512)
    # candidates
    ap.add_argument("--candidates", type=str, required=True)
    # output
    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg_all = json.load(f)
    cfg = cfg_all[args.model_name]

    accelerator, device_map = init_accelerator(args.gpu_id, cfg)
    device = accelerator.device

    with open(args.candidates) as f:
        candidates = json.load(f)
    print(f"# candidates: {len(candidates)}", flush=True)

    model_id = os.path.join(args.model_path, args.model_name)
    tokenizer = get_tokenizer(model_id, use_fast=True)

    # ----- Calibration loader -----
    print("Loading gov_report calibration...", flush=True)
    calib_loader = get_loader(
        args.calib_dataset, model=model_id,
        n_sample=args.n_sample, batch_size=args.data_batch_size,
        train=True, seed=args.seed,
        seqlen=args.seqlen, min_seqlen=args.min_seqlen,
    )
    calib_batches = list(calib_loader)
    print(f"calib batches: {len(calib_batches)} × seqlen up to {args.seqlen}", flush=True)

    # ----- RULER instances -----
    print(f"Building RULER instances for {args.ruler_tasks} ...", flush=True)
    ruler_instances = build_ruler_instances(
        tokenizer, model_id, args.ruler_tasks, args.ruler_n_per_task, args.ruler_length, args.seed
    )
    print(f"# ruler instances: {len(ruler_instances)}", flush=True)

    # ----- Dense model: compute key tokens on calib + cache nothing else -----
    print("Loading dense FP16 model for key-token discovery + outer pass...", flush=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else "auto"
    dense = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
    dense.eval()

    # Build per-batch key-token index lists. find_key_token works on the model's
    # own tokenization; we then map char intervals back via cal_overlap.
    print("Discovering key tokens on calibration batches...", flush=True)
    key_token_per_batch = []  # list[list[Tensor or None]] — outer over batches, inner over batch_size
    t0 = time.time()
    for b_idx, (inputs, attn, labels) in enumerate(calib_batches):
        bs = inputs.shape[0]
        per_seq = []
        for s_idx in range(bs):
            ids = inputs[s_idx:s_idx + 1]
            if attn is not None:
                m = attn[s_idx]
                actual = int(m.sum().item())
                ids = ids[:, :actual]
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            char_intervals = find_key_token(
                text, dense, tokenizer,
                trunc_len=args.trunc_len, sliding_window=args.sliding_window,
                save_path="", alpha=args.alpha, beta=args.beta,
            )
            try:
                enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
                offsets = enc["offset_mapping"][0]
            except NotImplementedError:
                offsets = compute_offsets(text, tokenizer, ids)[0]
            kt = cal_overlap(offsets, char_intervals.tolist() if hasattr(char_intervals, "tolist") else char_intervals)
            if kt is None or len(kt) == 0:
                per_seq.append(None)
            else:
                # cal_overlap returns indices into input tokens; convert to shift indices.
                shift_kt = [t - 1 for t in kt if t >= 1]
                per_seq.append(torch.tensor(shift_kt, dtype=torch.long) if shift_kt else None)
        key_token_per_batch.append(per_seq)
    print(f"Key-token discovery done in {time.time()-t0:.1f}s", flush=True)

    n_kt_total = sum(0 if x is None else len(x) for batch in key_token_per_batch for x in batch)
    print(f"Total key tokens across calib: {n_kt_total}", flush=True)
    with open(os.path.join(args.save, "key_token_counts.json"), "w") as f:
        json.dump({
            "n_total": n_kt_total,
            "per_batch": [[None if x is None else len(x) for x in batch] for batch in key_token_per_batch],
        }, f, indent=2)

    # ----- Build LlamaEvaluator (quant ensemble) -----
    # Use cross_entropy mode to skip its own dense loading / key-token loading.
    print("Building LlamaEvaluator for quant ensemble...", flush=True)
    method = {"w": args.w_method, "kv": args.kv_method}
    bits = {"w": list(args.w_bits), "k": [2, 3, 4], "v": [2, 3, 4]}  # only used to gate KV path
    group_size = {"w": args.w_group_size, "k": args.k_group_size, "v": args.v_group_size}
    evaluator = LlamaEvaluator(
        config=cfg,
        accelerator=accelerator,
        method=method,
        model_id=model_id,
        quant_model_paths=args.quant_model_paths,
        outlier=None,
        datasets=[args.calib_dataset],
        data_batch_size=args.data_batch_size,
        seed=args.seed,
        seqlen=args.seqlen,
        min_seqlen=args.min_seqlen,
        n_sample=args.n_sample,
        device_map=device_map,
        dtype=args.dtype,
        loss_func="cross_entropy",
        bits=bits,
        group_size=group_size,
        residual_length=args.residual_length,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        packing=False,
        use_key_token=False,
    )

    # ----- Outer loop: for each calib batch, dense fwd once, then all configs -----
    n_cand = len(candidates)
    cand_X_all = [[] for _ in range(n_cand)]
    cand_X_key = [[] for _ in range(n_cand)]
    cand_X_key_seqlens = [[] for _ in range(n_cand)]
    cand_X_all_seqlens = [[] for _ in range(n_cand)]

    print("=== Calibration outer loop ===", flush=True)
    for b_idx, (inputs, attn, labels) in enumerate(calib_batches):
        inputs = inputs.to(device)
        with torch.no_grad():
            dense_out = dense(inputs)
        dense_logits = dense_out.logits[:, :-1, :].contiguous()  # [B,T-1,V]
        del dense_out
        torch.cuda.empty_cache()

        for c_idx, arch in enumerate(candidates):
            t_c = time.time()
            qmodel = evaluator.sample(arch)
            qmodel.eval()
            q_full_logits = quant_forward_strided(qmodel, inputs, args.stride)
            q_logits = q_full_logits[:, :-1, :].contiguous()
            del q_full_logits

            # Per-sequence JSD aggregation.
            B = dense_logits.shape[0]
            for s_idx in range(B):
                d = dense_logits[s_idx]
                q = q_logits[s_idx]
                # All-token JSD mean.
                jsd_all = jsd_pair(d, q).mean().item()
                cand_X_all[c_idx].append(jsd_all)
                cand_X_all_seqlens[c_idx].append(d.shape[0])
                # Key-token JSD mean.
                kt = key_token_per_batch[b_idx][s_idx]
                if kt is not None and len(kt) > 0:
                    kt_dev = kt.to(d.device)
                    kt_dev = kt_dev[kt_dev < d.shape[0]]
                    if len(kt_dev) > 0:
                        jsd_k = jsd_pair(d[kt_dev], q[kt_dev]).mean().item()
                        cand_X_key[c_idx].append(jsd_k)
                        cand_X_key_seqlens[c_idx].append(len(kt_dev))

            del q_logits
            torch.cuda.empty_cache()
            if c_idx == 0 or (c_idx + 1) % 10 == 0:
                print(f"  batch {b_idx+1}/{len(calib_batches)} cand {c_idx+1}/{n_cand}: {time.time()-t_c:.1f}s", flush=True)

        del dense_logits
        torch.cuda.empty_cache()

    # ----- RULER outer loop -----
    cand_Y_per_task = [{t: [] for t in args.ruler_tasks} for _ in range(n_cand)]
    cand_Y_top1_per_task = [{t: [] for t in args.ruler_tasks} for _ in range(n_cand)]
    print("=== RULER outer loop ===", flush=True)
    for inst_i, inst in enumerate(ruler_instances):
        ids = inst["input_ids"].to(device)
        ans_idx = inst["answer_shift_idx"].to(device)
        with torch.no_grad():
            dense_out = dense(ids)
        d_logits = dense_out.logits[0, :-1, :]  # [T-1, V]
        del dense_out
        # Filter answer indices to valid range.
        ans_idx = ans_idx[ans_idx < d_logits.shape[0]]
        if len(ans_idx) == 0:
            continue
        d_ans = d_logits[ans_idx]
        d_top1 = d_ans.argmax(-1)

        for c_idx, arch in enumerate(candidates):
            qmodel = evaluator.sample(arch)
            qmodel.eval()
            q_full = quant_forward_strided(qmodel, ids, args.stride)
            q_logits = q_full[0, :-1, :]
            q_ans = q_logits[ans_idx]
            jsd_ans = jsd_pair(d_ans, q_ans).mean().item()
            top1_match = (q_ans.argmax(-1) == d_top1).float().mean().item()
            cand_Y_per_task[c_idx][inst["task"]].append(jsd_ans)
            cand_Y_top1_per_task[c_idx][inst["task"]].append(top1_match)
            del q_full, q_logits, q_ans
            torch.cuda.empty_cache()

        del d_logits, d_ans
        torch.cuda.empty_cache()
        if (inst_i + 1) % 10 == 0:
            print(f"  ruler instance {inst_i+1}/{len(ruler_instances)}", flush=True)

    # ----- Aggregate + save CSV -----
    rows = []
    for c_idx, arch in enumerate(candidates):
        # Token-weighted means.
        def wmean(vals, ws):
            if not vals:
                return float("nan")
            v = np.array(vals); w = np.array(ws, dtype=float)
            return float((v * w).sum() / w.sum()) if w.sum() > 0 else float("nan")

        x_all = wmean(cand_X_all[c_idx], cand_X_all_seqlens[c_idx])
        x_key = wmean(cand_X_key[c_idx], cand_X_key_seqlens[c_idx])

        row = {
            "config_id": c_idx,
            "X_all": x_all,
            "X_key": x_key,
        }
        all_Y = []
        for task in args.ruler_tasks:
            ys = cand_Y_per_task[c_idx][task]
            t1 = cand_Y_top1_per_task[c_idx][task]
            row[f"Y_jsd_{task}"] = float(np.mean(ys)) if ys else float("nan")
            row[f"Y_top1_{task}"] = float(np.mean(t1)) if t1 else float("nan")
            all_Y.extend(ys)
        row["Y_jsd_mean"] = float(np.mean(all_Y)) if all_Y else float("nan")
        # Bit summary.
        w_bits = []
        for ln, vals in arch["q"]["w"].items():
            w_bits.extend(vals)
        k_bits = [x[0] for x in arch["q"]["k"]]
        v_bits = [x[0] for x in arch["q"]["v"]]
        row["w_bits_mean"] = float(np.mean(w_bits))
        row["k_bits_mean"] = float(np.mean(k_bits))
        row["v_bits_mean"] = float(np.mean(v_bits))
        rows.append(row)

    csv_path = os.path.join(args.save, "results.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Wrote {csv_path}", flush=True)

    with open(os.path.join(args.save, "raw.json"), "w") as f:
        json.dump({
            "tasks": args.ruler_tasks,
            "X_all": cand_X_all,
            "X_key": cand_X_key,
            "Y_per_task": cand_Y_per_task,
            "Y_top1_per_task": cand_Y_top1_per_task,
        }, f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
