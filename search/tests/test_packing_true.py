"""
packing=True (KIVIDynamicCache) 경로 버그 검증 테스트

테스트 항목:
1. cuda_bmm_fA_qB_outer - KEY attention (MHA / GQA)
2. cuda_bmm_fA_qB_outer - VALUE attention (MHA / GQA, non-contiguous fA slice)
3. KIVIDynamicCache prefill + 다중 generation steps (lazy_update 포함)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import math


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed=0):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_kv_quant(B, n_kv, T, D, group_size, bits, device='cuda'):
    """key/value (B,n_kv,T,D) 를 quantize+pack하여 반환."""
    from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim
    k = torch.randn(B, n_kv, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, n_kv, T, D, device=device, dtype=torch.float16)
    # KEY: channel-wise → transpose to (B,n_kv,D,T) before pack
    kq, ks, km = triton_quantize_and_pack_along_last_dim(k.transpose(2,3).contiguous(), group_size, bits)
    # VALUE: token-wise → (B,n_kv,T,D) as-is
    vq, vs, vm = triton_quantize_and_pack_along_last_dim(v, group_size, bits)
    return k, kq, ks, km, v, vq, vs, vm


# ── Test 1: KEY GEMV (non-contiguous 없음) ────────────────────────────────────

def test_key_gemv_mha(bits=4, device='cuda'):
    """KEY GEMV: Q @ K^T, MHA (n_h == n_kv)"""
    from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer
    from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_kcache

    B, n_h, T, D = 2, 8, 256, 64
    group_size = 64
    set_seed(42)

    k = torch.randn(B, n_h, T, D, device=device, dtype=torch.float16)
    q = torch.randn(B, n_h, 1, D, device=device, dtype=torch.float16)

    kq, ks, km_val = triton_quantize_and_pack_along_last_dim(k.transpose(2,3).contiguous(), group_size, bits)

    out = cuda_bmm_fA_qB_outer(group_size, q, kq, ks, km_val, bits)
    # reference: dequantize then matmul
    k_deq = unpack_and_dequant_kcache(kq, ks.unsqueeze(-2), km_val.unsqueeze(-2), group_size, bits)
    ref = torch.matmul(q, k_deq.transpose(2, 3))

    err = (out - ref).abs().mean() / (ref.abs().mean() + 1e-5)
    print(f"[KEY MHA  bits={bits}] rel_err={err.item():.4f}", "PASS" if err < 0.05 else "FAIL")
    return err < 0.05


def test_key_gemv_gqa(bits=4, device='cuda'):
    """KEY GEMV: Q @ K^T, GQA (n_h > n_kv)"""
    from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer
    from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_kcache

    B, n_h, n_kv, T, D = 1, 32, 8, 128, 64
    group_size = 64
    set_seed(7)

    k = torch.randn(B, n_kv, T, D, device=device, dtype=torch.float16)
    q = torch.randn(B, n_h, 1, D, device=device, dtype=torch.float16)

    kq, ks, km_val = triton_quantize_and_pack_along_last_dim(k.transpose(2,3).contiguous(), group_size, bits)

    out = cuda_bmm_fA_qB_outer(group_size, q, kq, ks, km_val, bits)

    # reference: repeat kv then dequant matmul
    k_deq = unpack_and_dequant_kcache(kq, ks.unsqueeze(-2), km_val.unsqueeze(-2), group_size, bits)
    # repeat along n_kv dim
    n_rep = n_h // n_kv
    k_deq_rep = k_deq[:, :, None, :, :].expand(B, n_kv, n_rep, T, D).reshape(B, n_h, T, D)
    ref = torch.matmul(q, k_deq_rep.transpose(2, 3))

    err = (out - ref).abs().mean() / (ref.abs().mean() + 1e-5)
    print(f"[KEY GQA  bits={bits}] rel_err={err.item():.4f}", "PASS" if err < 0.05 else "FAIL")
    return err < 0.05


# ── Test 2: VALUE GEMV (non-contiguous fA slice - 핵심 버그 케이스) ────────────

def test_value_gemv_noncontiguous(bits=4, device='cuda'):
    """VALUE GEMV: attn_weights @ V

    핵심: attn_weights[:, :, :, :-value_full_length] 는 non-contiguous slice.
    fA.view() → fA.reshape() 수정 없이는 RuntimeError 발생.
    """
    from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer
    from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim

    B, n_h, n_kv, T_quant, T_full, D = 1, 32, 8, 128, 32, 64
    group_size = 64
    set_seed(99)

    T_total = T_quant + T_full
    v = torch.randn(B, n_kv, T_quant, D, device=device, dtype=torch.float16)
    vq, vs, vm = triton_quantize_and_pack_along_last_dim(v, group_size, bits)

    # attn_weights 전체 (contiguous)
    attn_weights = torch.randn(B, n_h, 1, T_total, device=device, dtype=torch.float16)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # non-contiguous slice: 핵심 버그 케이스
    fA_slice = attn_weights[:, :, :, :-T_full]   # non-contiguous!
    assert not fA_slice.is_contiguous(), "slice should be non-contiguous"

    try:
        out = cuda_bmm_fA_qB_outer(group_size, fA_slice, vq, vs, vm, bits)
    except RuntimeError as e:
        print(f"[VALUE GQA bits={bits}] FAIL (RuntimeError): {e}")
        return False

    # reference: dequantize then matmul
    from quant.kivi_utils.new_pack import unpack_and_dequant_vcache
    v_deq = unpack_and_dequant_vcache(vq, vs.unsqueeze(-1), vm.unsqueeze(-1), group_size, bits)
    n_rep = n_h // n_kv
    v_deq_rep = v_deq[:, :, None, :, :].expand(B, n_kv, n_rep, T_quant, D).reshape(B, n_h, T_quant, D)
    w_quant = attn_weights[:, :, :, :T_quant]
    ref = torch.matmul(w_quant, v_deq_rep)

    err = (out - ref).abs().mean() / (ref.abs().mean() + 1e-5)
    print(f"[VALUE GQA bits={bits}] rel_err={err.item():.4f}", "PASS" if err < 0.05 else "FAIL")
    return err < 0.05


# ── Test 3: KIVIDynamicCache end-to-end ──────────────────────────────────────

def test_kivi_dynamic_cache_e2e(bits=4, device='cuda'):
    """KIVIDynamicCache prefill → 여러 generation steps (lazy_update 포함)

    packing=True 와 packing=False (fake-quant) 의 attention output 을 비교.
    두 경로는 동일한 양자화 오차를 가지므로 결과가 유사해야 함.
    """
    from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache

    B, n_h, n_kv, D = 1, 8, 8, 64
    residual_length = 64
    group_size = 64
    T_prefill = 128    # 2 × residual_length → 전량 quant
    n_gen = 80         # lazy_update 가 여러번 발생
    set_seed(11)

    cfg_pack = KIVICacheConfig(
        k_bits=[bits] * 4,
        v_bits=[bits] * 4,
        k_group_size=[group_size] * 4,
        v_group_size=[group_size] * 4,
        k_quant_scheme='channel',
        v_quant_scheme='token',
        residual_length=residual_length,
        packing=True,
    )
    cfg_fake = KIVICacheConfig(
        k_bits=[bits] * 4,
        v_bits=[bits] * 4,
        k_group_size=[group_size] * 4,
        v_group_size=[group_size] * 4,
        k_quant_scheme='channel',
        v_quant_scheme='token',
        residual_length=residual_length,
        packing=False,
    )

    layer_idx = 0
    k_prefill = torch.randn(B, n_kv, T_prefill, D, device=device, dtype=torch.float16)
    v_prefill = torch.randn(B, n_kv, T_prefill, D, device=device, dtype=torch.float16)

    # ---- prefill both caches ----
    cache_pack = KIVIDynamicCache(cfg_pack)
    cache_fake = KIVIFakeCache(cfg_fake)
    # KIVIDynamicCache.update() 는 km(mean) 을 내부적으로 뺌
    cache_pack.update(k_prefill.clone(), v_prefill.clone(), layer_idx, {})
    cache_fake.update(k_prefill.clone(), v_prefill.clone(), layer_idx, {})

    # ---- generation steps ----
    from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer
    from quant.kivi_utils.new_pack import unpack_and_dequant_kcache, unpack_and_dequant_vcache
    from model.kivi_utils import repeat_kv

    errors_k, errors_v = [], []

    for step in range(n_gen):
        q = torch.randn(B, n_h, 1, D, device=device, dtype=torch.float16)
        k_new = torch.randn(B, n_kv, 1, D, device=device, dtype=torch.float16)
        v_new = torch.randn(B, n_kv, 1, D, device=device, dtype=torch.float16)

        # update both caches
        cache_pack.update(k_new.clone(), v_new.clone(), layer_idx, {})
        cache_fake.update(k_new.clone(), v_new.clone(), layer_idx, {})

        # ---- packing=True path ----
        kq_trans = cache_pack.key_states_quant_trans_cache[layer_idx]
        ks_trans = cache_pack.key_scale_trans_cache[layer_idx]
        km_trans = cache_pack.key_mn_trans_cache[layer_idx]
        k_full  = cache_pack.key_states_full_cache[layer_idx]
        vq      = cache_pack.value_states_quant_cache[layer_idx]
        vs_     = cache_pack.value_scale_cache[layer_idx]
        vm_     = cache_pack.value_mn_cache[layer_idx]
        v_full  = cache_pack.value_states_full_cache[layer_idx]

        # KEY attention
        if kq_trans is not None:
            att_qk_quant = cuda_bmm_fA_qB_outer(group_size, q, kq_trans, ks_trans, km_trans, bits)
        else:
            att_qk_quant = None

        att_qk_full = torch.matmul(q, repeat_kv(k_full, n_h // n_kv).transpose(2, 3))
        if att_qk_quant is not None:
            att_pack = torch.cat([att_qk_quant, att_qk_full], dim=-1) / math.sqrt(D)
        else:
            att_pack = att_qk_full / math.sqrt(D)
        att_pack = torch.softmax(att_pack, dim=-1)

        # VALUE attention (핵심: non-contiguous slice)
        v_full_len = v_full.shape[-2]
        if vq is not None:
            v_out_pack = cuda_bmm_fA_qB_outer(
                group_size,
                att_pack[:, :, :, :-v_full_len],   # non-contiguous slice
                vq, vs_, vm_, bits
            )
            v_out_pack += torch.matmul(att_pack[:, :, :, -v_full_len:], repeat_kv(v_full, n_h // n_kv))
        else:
            v_out_pack = torch.matmul(att_pack, repeat_kv(v_full, n_h // n_kv))

        # lazy_update
        cache_pack.lazy_update(layer_idx)

        # ---- packing=False path (reference) ----
        k_fake_all = torch.cat([
            cache_fake.key_states_quant_trans_cache[layer_idx],
            cache_fake.key_states_full_cache[layer_idx]
        ], dim=-2) if cache_fake.key_states_quant_trans_cache[layer_idx] is not None else cache_fake.key_states_full_cache[layer_idx]
        v_fake_all = torch.cat([
            cache_fake.value_states_quant_cache[layer_idx],
            cache_fake.value_states_full_cache[layer_idx]
        ], dim=-2) if cache_fake.value_states_quant_cache[layer_idx] is not None else cache_fake.value_states_full_cache[layer_idx]

        att_fake = torch.softmax(
            torch.matmul(q, repeat_kv(k_fake_all, n_h // n_kv).transpose(2, 3)) / math.sqrt(D),
            dim=-1
        )
        v_out_fake = torch.matmul(att_fake, repeat_kv(v_fake_all, n_h // n_kv))

        cache_fake.lazy_update(layer_idx)

        # ---- 오차 계산 ----
        err_v = (v_out_pack - v_out_fake).abs().mean() / (v_out_fake.abs().mean() + 1e-5)
        errors_v.append(err_v.item())

    avg_err = sum(errors_v) / len(errors_v)
    max_err = max(errors_v)
    print(f"[E2E  bits={bits}] avg_rel_err={avg_err:.4f}  max_rel_err={max_err:.4f}",
          "PASS" if avg_err < 0.15 else "FAIL")
    return avg_err < 0.15


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("packing=True 버그 검증 테스트")
    print("=" * 60)

    results = []

    for bits in [2, 4, 8]:
        results.append(test_key_gemv_mha(bits=bits))
        results.append(test_key_gemv_gqa(bits=bits))
        results.append(test_value_gemv_noncontiguous(bits=bits))

    results.append(test_kivi_dynamic_cache_e2e(bits=4))

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"결과: {passed}/{total} PASS")
    if passed < total:
        print("일부 테스트 실패 — 추가 수정 필요")
    else:
        print("모든 테스트 통과!")
