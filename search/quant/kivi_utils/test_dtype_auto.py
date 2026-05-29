"""Smoke-test: --dtype auto across the LlamaEvaluator + KIVI path.

Verifies:
  1. process_dtype('auto')                       → 'auto' string
  2. LlamaEvaluator(dtype='auto', method=hqq…)
     after load → self.dtype is a concrete torch.dtype (resolved from
     the HQQ pre-quant config's torch_dtype)
  3. KIVI cache, after prefill on that model, stores scale/min in the
     same torch.dtype the model uses (no silent fp16/bf16 mismatch)

Lightweight: skips the dataset/get_logits machinery by using
loss_func='cross_entropy' (no dense-logits precompute) and skipping the
JSD path. Runs only on a tiny synthetic prefill — no actual eval loop.
"""
import os, sys, json, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from accelerate import Accelerator  # noqa: E402
from evaluator import LlamaEvaluator  # noqa: E402
from utils.func import process_dtype, init_accelerator  # noqa: E402

# Reuse the e2e test's tiny KIVIDynamicCache exercise after we sample the
# uniform arch.

MODEL_PATH = "/SSD/huggingface/meta-llama"
MODEL_NAME = "Llama-3.1-8B-Instruct"
# fp16-baked HQQ dir → expects self.dtype to resolve to torch.float16.
# bf16-baked HQQ dir → expects self.dtype to resolve to torch.bfloat16.
HQQ_FP16 = "/SSD/hqq/Llama-3.1-8B-Instruct_3bit_128gs_1axis_float16"
HQQ_BF16 = "/SSD/hqq/Llama-3.1-8B-Instruct_3bit_128gs_1axis_bfloat16"


def _build_uniform_arch(config, n_block, w_bits=3, k_bits=2, k_gs=32, v_bits=2, v_gs=32):
    return {
        'q': {
            'w': {linear: [w_bits] * n_block for linear in config['linear']},
            'k': [[k_bits, k_gs]] * n_block,
            'v': [[v_bits, v_gs]] * n_block,
        },
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }


def test_process_dtype():
    print("\n[1] process_dtype")
    assert process_dtype('auto') == 'auto'
    assert process_dtype('float16') is torch.float16
    assert process_dtype('bfloat16') is torch.bfloat16
    print("  OK  process_dtype passes through 'auto' and maps fp16/bf16")


def _check_one(hqq_path, expect_dtype, label):
    print(f"\n[2/3] LlamaEvaluator(dtype='auto')  HQQ={os.path.basename(hqq_path)}  → expect {expect_dtype}")
    with open(os.path.join(ROOT, 'config/llama.json')) as f:
        config = json.load(f)[MODEL_NAME]
    n_block = int(config['n_block'])
    accelerator, device_map = init_accelerator('0', config)

    ev = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        method={'w': 'hqq', 'kv': ['kivi']},
        model_id=os.path.join(MODEL_PATH, MODEL_NAME),
        quant_model_paths=[hqq_path],
        datasets=['wikitext2'],
        data_batch_size=1, seed=0,
        seqlen=2048, n_sample=2,
        device_map=device_map,
        dtype='auto',                       # ← key: pass 'auto'
        loss_func='cross_entropy',
        bits={'w': [3], 'k': [2], 'v': [2]},
        group_size={'w': 128, 'k': [32], 'v': [32]},
        residual_length=128,
        k_quant_scheme='channel', v_quant_scheme='token', packing=True,
    )

    assert isinstance(ev.dtype, torch.dtype), \
        f"evaluator.dtype not resolved: {ev.dtype!r}"
    assert ev.dtype is expect_dtype, \
        f"{label}: evaluator.dtype={ev.dtype} but expected {expect_dtype}"
    print(f"  OK  evaluator.dtype resolved to {ev.dtype}")

    # Sample the uniform arch → triggers HQQ-weight re-bind + sets kivi_config.
    arch = _build_uniform_arch(config, n_block)
    model = ev.sample(arch)
    actual_param = next(model.parameters()).dtype
    print(f"  model param dtype = {actual_param}")
    assert actual_param is expect_dtype, \
        f"model params {actual_param} ≠ expected {expect_dtype}"

    # ---- KIVI dtype propagation ----
    # Build a single decoder block's prefill K/V (matching the model dtype)
    # and run it through KIVIDynamicCache to confirm the stored scale/min
    # also lands in expect_dtype (i.e. KIVI follows model dtype, not fp16).
    from model.KIVICache import KIVICacheConfig, KIVIDynamicCache
    cfg = KIVICacheConfig(
        k_bits=[2] * n_block, v_bits=[2] * n_block,
        k_group_size=[32] * n_block, v_group_size=[32] * n_block,
        residual_length=128, packing=True,
    )
    cache = KIVIDynamicCache(cfg)
    B, n_kv, T, D = 1, 8, 256, 128  # any T multiple of gs+residual works
    K = torch.randn(B, n_kv, T, D, device='cuda', dtype=expect_dtype)
    V = torch.randn(B, n_kv, T, D, device='cuda', dtype=expect_dtype)
    cache.update(K.clone(), V.clone(), layer_idx=0)

    sc = cache.key_scale_trans_cache[0]
    mn = cache.value_mn_cache[0]
    assert sc.dtype is expect_dtype, f"KIVI K-scale dtype {sc.dtype} ≠ {expect_dtype}"
    assert mn.dtype is expect_dtype, f"KIVI V-mn   dtype {mn.dtype} ≠ {expect_dtype}"
    print(f"  OK  KIVI packed scale/min dtype = {expect_dtype}")

    del ev, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    test_process_dtype()
    _check_one(HQQ_FP16, torch.float16, "fp16 HQQ")
    _check_one(HQQ_BF16, torch.bfloat16, "bf16 HQQ")
    print("\nAll --dtype auto checks PASSED.")
