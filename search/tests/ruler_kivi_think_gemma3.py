"""RULER niah_multikey_3 @ 16384, nsample=50 — Gemma3-12B KIVI/ThinK.

Long-context retrieval stress test of the KIVI cache path (prefill 16k +
greedy decode through KIVIFakeCache) for gemma-3-12b-it. Compares FP16-KV
baseline vs KIVI vs KIVI+ThinK.

Run:
  CUDA_VISIBLE_DEVICES=0 python tests/ruler_kivi_think_gemma3.py
"""
import sys, gc, json, traceback, torch
sys.path.insert(0, "/NAS/SJ/actquant/search")

from transformers import AutoConfig, AutoTokenizer, Gemma3ForCausalLM
from model.replace import replace_kv_cache
from utils.ruler import eval_ruler

MODEL = "/SSD/huggingface/google/gemma-3-12b-it"
N_BLOCK = 48
DTYPE = torch.bfloat16
RULER_YAML = "/NAS/SJ/actquant/search/utils/ruler_utils"
TASK = "niah_multikey_3"
LEN = 16384
NSAMPLE = 50

TOK = AutoTokenizer.from_pretrained(MODEL)


def load_fp16():
    cfg = AutoConfig.from_pretrained(MODEL)
    m = Gemma3ForCausalLM.from_pretrained(
        MODEL, config=cfg.text_config, torch_dtype=DTYPE,
        device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    m.eval()
    return m


def set_kivi_arch(model, kbits, vbits, kgs, vgs, residual_length=128,
                  k_prune=0, v_prune=0):
    c = model.config.kivi_config
    c.k_bits = [kbits] * N_BLOCK
    c.v_bits = [vbits] * N_BLOCK
    c.k_group_size = [kgs] * N_BLOCK
    c.v_group_size = [vgs] * N_BLOCK
    c.residual_length = residual_length
    c.k_pruning_dim = [k_prune] * N_BLOCK
    c.v_pruning_dim = [v_prune] * N_BLOCK


def run_ruler(model, tag):
    model.config.use_cache = True
    rp = f"/tmp/ruler_g3_{tag}_{TASK}_{LEN}.json"
    try:
        eval_ruler(model, tokenizer=TOK, model_id=MODEL, yaml_path=RULER_YAML,
                   tasks=[TASK], length=[LEN], batch_size=1, nsample=NSAMPLE,
                   seed=0, gen_toks=128, result_path=rp)
        score = json.load(open(rp)).get(TASK)
    except Exception:
        print(f"[RULER {tag}] FAILED:\n{traceback.format_exc()}", flush=True)
        score = None
    print(f"[RULER {tag}] {TASK}@{LEN} (n={NSAMPLE}) = {score}", flush=True)
    return score


def case(tag, methods=None, **kw):
    print(f"\n=== {tag} ===", flush=True)
    m = load_fp16()
    if methods is not None:
        m = replace_kv_cache(model=m, tokenizer=TOK, method=methods,
                             n_block=N_BLOCK, k_quant_scheme="channel",
                             v_quant_scheme="token", residual_length=128,
                             packing=False, quant_kv_output=False,
                             k_pruning_dim=kw.get("k_prune", 0),
                             v_pruning_dim=kw.get("v_prune", 0))
        set_kivi_arch(m, kbits=kw["kbits"], vbits=kw["vbits"], kgs=128, vgs=128,
                      residual_length=128, k_prune=kw.get("k_prune", 0),
                      v_prune=kw.get("v_prune", 0))
        m.config.quant_kv_output = False
    s = run_ruler(m, tag)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return s


def main():
    r = {}
    r["fp16"] = 0.64        # known baseline (native HybridCache)
    r["kivi_k4v4"] = 0.60   # known (verified post sliding-mask fix)
    r["think_kp64"] = 0.60  # known (verified post sliding-mask fix)
    # Verify V-pruning in the long-context decode path
    r["think_vp64"] = case("KIVI+ThinK k4v4 vp64", methods=["kivi", "think"],
                           kbits=4, vbits=4, k_prune=0, v_prune=64)
    r["think_kp64_vp64"] = case("KIVI+ThinK k4v4 kp64 vp64",
                                methods=["kivi", "think"],
                                kbits=4, vbits=4, k_prune=64, v_prune=64)
    print("\n================ SUMMARY ================")
    for k, v in r.items():
        print(f"{TASK}@{LEN} n={NSAMPLE}  {k:>22} = {v}")


if __name__ == "__main__":
    main()
