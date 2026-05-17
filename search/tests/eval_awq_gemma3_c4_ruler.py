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
RULER_LEN, RULER_NSAMPLE = 4096, 5


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


def _force_eager_attn(model):
    # Gemma3's sliding-window mask hits "p.attn_bias_ptr is not correctly
    # aligned" in the efficient-attention kernels during KV-cache generation.
    # Eager attention has no alignment constraint.
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass
    model.config._attn_implementation = "eager"
    for mod in model.modules():
        if hasattr(mod, "config"):
            mod.config._attn_implementation = "eager"
        if hasattr(mod, "_attn_implementation"):
            mod._attn_implementation = "eager"


def run_ruler(model, tag):
    from utils.ruler import eval_ruler
    rp = f"/tmp/ruler_g3_{tag}.json"
    model.config.use_cache = True
    _force_eager_attn(model)
    try:
        eval_ruler(model, tokenizer=EVAL_TOK, model_id=MODEL,
                   yaml_path=RULER_YAML, tasks=["niah_single_1"],
                   length=[RULER_LEN], batch_size=1, nsample=RULER_NSAMPLE,
                   seed=0, gen_toks=128, result_path=rp)
        return json.load(open(rp))
    except Exception:
        print(f"[RULER {tag}] FAILED:\n{traceback.format_exc()}")
        return None


def main():
    res = {}
    m = AWQ(model_name=MODEL, config={}, arch=arch, device_map={"": 0},
            group_size=128, dtype=torch.bfloat16, clip_asym=True)

    m.model = m.model.to("cuda:0")
    res["fp16_c4_ppl"], nseg = c4_ppl(m.model)
    print(f"[FP16] C4 ppl = {res['fp16_c4_ppl']:.4f}  (n_seg={nseg})")
    res["fp16_ruler"] = run_ruler(m.model, "fp16")
    print(f"[FP16] RULER niah_single_1 = {res['fp16_ruler']}")
    m.model = m.model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    m.run(nsamples=8, seqlen=512)            # identical calibration to smoke

    res["awq_c4_ppl"], _ = c4_ppl(m.model)
    print(f"[AWQ4] C4 ppl = {res['awq_c4_ppl']:.4f}")
    res["awq_ruler"] = run_ruler(m.model, "awq")
    print(f"[AWQ4] RULER niah_single_1 = {res['awq_ruler']}")

    r = res["awq_c4_ppl"] / res["fp16_c4_ppl"]
    print("\n================ SUMMARY ================")
    print(f"C4 ppl   FP16={res['fp16_c4_ppl']:.3f}  AWQ4={res['awq_c4_ppl']:.3f}"
          f"  ratio={r:.3f}")
    if res["fp16_ruler"] and res["awq_ruler"]:
        f = res["fp16_ruler"].get("niah_single_1")
        a = res["awq_ruler"].get("niah_single_1")
        print(f"RULER niah_single_1@{RULER_LEN}  FP16={f}  AWQ4={a}")
    json.dump(res, open("/tmp/eval_g3_summary.json", "w"), indent=2, default=str)
    print("DONE")


if __name__ == "__main__":
    main()
