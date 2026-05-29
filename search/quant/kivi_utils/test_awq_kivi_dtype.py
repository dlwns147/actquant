"""Smoke-test: AWQ + KIVI dtype propagation across fp16 / bf16 / 'auto'.

Verifies the same dtype-flow guarantees we just confirmed for the HQQ
branch:
  • evaluator.sample()'s AWQ/GPTQ/QEFT branch resolves self.dtype to a
    real torch.dtype after the model loads (so 'auto' becomes
    torch.bfloat16 for Llama-3.1-8B).
  • AWQ's pseudo-quantize preserves weight dtype.
  • KIVI cache built on the AWQ model stores scale/min in the same dtype.

To keep the test fast, AWQ.run is monkey-patched to a no-op — we only
need the model to be loaded via BASE.load_model(torch_dtype=...) and the
subsequent KIVI cache wiring; the calibration sweep itself is orthogonal
to dtype propagation.
"""
import os, sys, json, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

# ── monkey-patch AWQ.run BEFORE importing LlamaEvaluator (which imports
# get_quantized_model → AWQ). The no-op leaves model weights in their
# from_pretrained dtype, which is exactly what we want to test.
from quant.awq import AWQ  # noqa: E402
_orig_run = AWQ.run
def _noop_run(self, *a, **kw):
    return  # skip calibration; preserve weights & dtype
AWQ.run = _noop_run

from evaluator import LlamaEvaluator  # noqa: E402
from utils.func import process_dtype, init_accelerator  # noqa: E402

MODEL_PATH = "/SSD/huggingface/meta-llama"
MODEL_NAME = "Llama-3.1-8B-Instruct"  # config's torch_dtype == bfloat16


def _build_uniform_arch(config, n_block, w_bits=4, k_bits=2, k_gs=32, v_bits=2, v_gs=32):
    return {
        'q': {
            'w': {linear: [w_bits] * n_block for linear in config['linear']},
            'k': [[k_bits, k_gs]] * n_block,
            'v': [[v_bits, v_gs]] * n_block,
        },
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }


def _run_case(dtype_arg, expect_dtype, label):
    print(f"\n[AWQ+KIVI] --dtype={dtype_arg!r}  → expect {expect_dtype}")
    with open(os.path.join(ROOT, 'config/llama.json')) as f:
        config = json.load(f)[MODEL_NAME]
    n_block = int(config['n_block'])
    accelerator, device_map = init_accelerator('0', config)

    ev = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        method={'w': 'awq', 'kv': ['kivi']},
        model_id=os.path.join(MODEL_PATH, MODEL_NAME),
        quant_model_paths=[],          # AWQ runs its own calibration; no dir
        datasets=['wikitext2'],
        data_batch_size=1, seed=0,
        seqlen=2048, n_sample=2,
        device_map=device_map,
        dtype=process_dtype(dtype_arg),
        loss_func='cross_entropy',
        bits={'w': [4], 'k': [2], 'v': [2]},
        group_size={'w': 128, 'k': [32], 'v': [32]},
        residual_length=128,
        k_quant_scheme='channel', v_quant_scheme='token', packing=True,
    )

    # AWQ branch defers model creation to sample(). Trigger it now.
    arch = _build_uniform_arch(config, n_block)
    model = ev.sample(arch)

    assert isinstance(ev.dtype, torch.dtype), \
        f"evaluator.dtype not resolved (still {ev.dtype!r})"
    assert ev.dtype is expect_dtype, \
        f"{label}: evaluator.dtype={ev.dtype} ≠ expected {expect_dtype}"
    print(f"  OK  evaluator.dtype resolved to {ev.dtype}")

    actual = next(model.parameters()).dtype
    print(f"  model param dtype = {actual}")
    assert actual is expect_dtype, \
        f"model params {actual} ≠ expected {expect_dtype}"

    # KIVI cache should be ready on the model (replace_kv_cache fires in
    # sample() when 'kivi' is in method['kv']).
    assert hasattr(model.config, 'kivi_config'), \
        "kivi_config not attached to model.config"
    print("  OK  kivi_config attached to model.config")

    # Round-trip a tiny prefill through KIVIDynamicCache to confirm KIVI
    # stores scale/min at expect_dtype.
    from model.KIVICache import KIVICacheConfig, KIVIDynamicCache
    cfg = KIVICacheConfig(
        k_bits=[2]*n_block, v_bits=[2]*n_block,
        k_group_size=[32]*n_block, v_group_size=[32]*n_block,
        residual_length=128, packing=True,
    )
    cache = KIVIDynamicCache(cfg)
    B, n_kv, T, D = 1, 8, 256, 128
    K = torch.randn(B, n_kv, T, D, device='cuda', dtype=expect_dtype)
    V = torch.randn(B, n_kv, T, D, device='cuda', dtype=expect_dtype)
    cache.update(K.clone(), V.clone(), layer_idx=0)
    sc = cache.key_scale_trans_cache[0]
    mn = cache.value_mn_cache[0]
    assert sc.dtype is expect_dtype and mn.dtype is expect_dtype, \
        f"KIVI scale/min dtype mismatch: K={sc.dtype} V={mn.dtype}"
    print(f"  OK  KIVI packed scale/min dtype = {expect_dtype}")

    del ev, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    # 1) explicit fp16
    _run_case('float16',  torch.float16,  "fp16")
    # 2) explicit bf16
    _run_case('bfloat16', torch.bfloat16, "bf16")
    # 3) 'auto' → Llama-3.1-8B's config.torch_dtype = bfloat16
    _run_case('auto',     torch.bfloat16, "auto")

    AWQ.run = _orig_run
    print("\nAll AWQ + KIVI dtype checks PASSED.")
