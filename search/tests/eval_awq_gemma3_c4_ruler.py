"""FP16 vs AWQ-4bit (Gemma3-12B) — C4 perplexity and RULER niah_single_1.

Calibration is kept identical to the smoke test (8 pileval samples, seqlen
512, pure 4-bit, group_size 128). RULER is best-effort: a failure there is
caught so the C4 numbers are still reported.
"""
import sys, gc, json, traceback, torch
sys.path.insert(0, "/NAS/SJ/actquant/search")
from quant.awq import AWQ
from transformers import AutoTokenizer

MODEL = "/SSD/huggingface/google/gemma-3-12b-it"
N_BLOCK = 48
LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
arch = {l: [4] * N_BLOCK for l in LINEARS}

EVAL_TOK = AutoTokenizer.from_pretrained(MODEL)
RULER_YAML = "/NAS/SJ/actquant/search/utils/ruler_utils"
RULER_LENS, RULER_NSAMPLE = [4096, 16384], 5


@torch.no_grad()
def c4_ppl(model, n_seg=40, seqlen=2048, n_docs=2000):
    from datasets import load_dataset
    data = load_dataset("allenai/c4",
                         data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                         split="validation")
    text = "\n\n".join(data[i]["text"] for i in range(min(n_docs, len(data))))
    enc = EVAL_TOK(text, return_tensors="pt").input_ids
    dev = next(model.parameters()).device
    nlls, n = [], min(n_seg, enc.shape[1] // seqlen)
    for i in range(n):
        ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
        nlls.append(model(ids, labels=ids).loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item(), n


@torch.no_grad()
def wikitext2_ppl(model, seqlen=2048):
    # standard GPTQ/AWQ wikitext2 protocol, seqlen 2048 (same as c4_ppl, so the
    # two ratios are apples-to-apples; the earlier 1.62 was a seqlen-512 smoke)
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = EVAL_TOK("\n\n".join(data["text"]), return_tensors="pt").input_ids
    dev = next(model.parameters()).device
    n = enc.shape[1] // seqlen
    nlls = []
    for i in range(n):
        ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
        nlls.append(model(ids, labels=ids).loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item(), n


def _set_fa2(model):
    # flash_attention_2 is the correct path for Gemma3 generation: it handles
    # sliding-window natively via window flags (no arbitrary additive attn_bias)
    # so it sidesteps the PyTorch SDPA mem-efficient "p.attn_bias_ptr is not
    # correctly aligned" error at odd seq lengths, and is O(n) memory so RULER
    # scales to long context. Gemma3Attention reads config._attn_implementation
    # per forward, so mutating the (shared) config switches dispatch.
    try:
        model.set_attn_implementation("flash_attention_2")
    except Exception:
        pass
    _set_attn(model, "flash_attention_2")


def _set_attn(model, impl):
    model.config._attn_implementation = impl
    for mod in model.modules():
        if hasattr(mod, "config"):
            mod.config._attn_implementation = impl
        if hasattr(mod, "_attn_implementation"):
            mod._attn_implementation = impl


def run_ruler(model, tag):
    from utils.ruler import eval_ruler
    model.config.use_cache = True
    # Snapshot so the FA2 switch does NOT leak into a later AWQ calibration
    # forward (awq.run calls run_awq on this same model object before its
    # internal reload). Calibration must stay on the default backend so PPL
    # is reproducible and independent of RULER ordering.
    orig_impl = model.config._attn_implementation
    _set_fa2(model)
    out = {}
    try:
        for L in RULER_LENS:
            rp = f"/tmp/ruler_g3_{tag}_{L}.json"
            try:
                eval_ruler(model, tokenizer=EVAL_TOK, model_id=MODEL,
                           yaml_path=RULER_YAML, tasks=["niah_single_1"],
                           length=[L], batch_size=1, nsample=RULER_NSAMPLE,
                           seed=0, gen_toks=128, result_path=rp)
                out[L] = json.load(open(rp)).get("niah_single_1")
            except Exception:
                print(f"[RULER {tag}@{L}] FAILED:\n{traceback.format_exc()}")
                out[L] = None
    finally:
        _set_attn(model, orig_impl)   # restore for the AWQ calibration that follows
    return out


def main():
    res = {}
    m = AWQ(model_name=MODEL, config={}, arch=arch, device_map={"": 0},
            group_size=128, dtype=torch.bfloat16, clip_asym=True)

    m.model = m.model.to("cuda:0")
    res["fp16_c4_ppl"], nseg = c4_ppl(m.model)
    print(f"[FP16] C4 ppl = {res['fp16_c4_ppl']:.4f}  (n_seg={nseg})")
    res["fp16_wikitext2_ppl"], nwk = wikitext2_ppl(m.model)
    print(f"[FP16] wikitext2 ppl = {res['fp16_wikitext2_ppl']:.4f}  (n_seg={nwk})")
    res["fp16_ruler"] = run_ruler(m.model, "fp16")
    print(f"[FP16] RULER niah_single_1 = {res['fp16_ruler']}")
    m.model = m.model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    m.run(nsamples=8, seqlen=512)            # identical calibration to smoke

    res["awq_c4_ppl"], _ = c4_ppl(m.model)
    print(f"[AWQ4] C4 ppl = {res['awq_c4_ppl']:.4f}")
    res["awq_wikitext2_ppl"], _ = wikitext2_ppl(m.model)
    print(f"[AWQ4] wikitext2 ppl = {res['awq_wikitext2_ppl']:.4f}")
    res["awq_ruler"] = run_ruler(m.model, "awq")
    print(f"[AWQ4] RULER niah_single_1 = {res['awq_ruler']}")

    rc = res["awq_c4_ppl"] / res["fp16_c4_ppl"]
    rw = res["awq_wikitext2_ppl"] / res["fp16_wikitext2_ppl"]
    print("\n================ SUMMARY ================")
    print(f"C4 ppl        FP16={res['fp16_c4_ppl']:.3f}  "
          f"AWQ4={res['awq_c4_ppl']:.3f}  ratio={rc:.3f}")
    print(f"wikitext2 ppl FP16={res['fp16_wikitext2_ppl']:.3f}  "
          f"AWQ4={res['awq_wikitext2_ppl']:.3f}  ratio={rw:.3f}  (seqlen 2048)")
    if res["fp16_ruler"] and res["awq_ruler"]:
        for L in RULER_LENS:
            print(f"RULER niah_single_1@{L} (FA2)  "
                  f"FP16={res['fp16_ruler'].get(L)}  "
                  f"AWQ4={res['awq_ruler'].get(L)}")
    json.dump(res, open("/tmp/eval_g3_summary.json", "w"), indent=2, default=str)
    print("DONE")


if __name__ == "__main__":
    main()
