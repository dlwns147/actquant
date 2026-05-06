"""Sanity check: quant_kv_output should now leave the last R tokens of K/V
untouched when residual_length > 0, and behave like before when R == 0."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.kivi_utils import quant_kv_output  # noqa: E402


class _Cfg:
    pass


def make_module(R, kbits=2, vbits=2, kgs=128, vgs=128, n_layers=1):
    m = type("M", (), {})()
    cfg = _Cfg()
    cfg.kivi_config = _Cfg()
    cfg.kivi_config.k_bits = [kbits] * n_layers
    cfg.kivi_config.v_bits = [vbits] * n_layers
    cfg.kivi_config.k_group_size = [kgs] * n_layers
    cfg.kivi_config.v_group_size = [vgs] * n_layers
    cfg.kivi_config.k_quant_scheme = "channel"
    cfg.kivi_config.v_quant_scheme = "token"
    cfg.kivi_config.residual_length = R
    cfg.kivi_config.enable_think = False
    cfg.kivi_config.k_pruning_dim = [0] * n_layers
    cfg.kivi_config.v_pruning_dim = [0] * n_layers
    cfg.quant_kv_output = True
    m.config = cfg
    m.layer_idx = 0
    return m


def main():
    torch.manual_seed(0)
    B, H, T, D = 1, 8, 2176, 128  # T = prompt_len + answer_len
    K = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")

    # ── Case 1: R=128, expect last 128 K/V tokens identical, first 2048 quantised ──
    m = make_module(R=128, kbits=2, vbits=2)
    Kq, Vq = quant_kv_output(m, K.clone(), V.clone(), attention_mask=None)
    last_k_diff = (Kq[:, :, -128:, :] - K[:, :, -128:, :]).abs().max().item()
    last_v_diff = (Vq[:, :, -128:, :] - V[:, :, -128:, :]).abs().max().item()
    first_k_diff = (Kq[:, :, :-128, :] - K[:, :, :-128, :]).abs().max().item()
    first_v_diff = (Vq[:, :, :-128, :] - V[:, :, :-128, :]).abs().max().item()
    print(f"[R=128] last-{128} K diff = {last_k_diff:.6e}  (must be 0)")
    print(f"[R=128] last-{128} V diff = {last_v_diff:.6e}  (must be 0)")
    print(f"[R=128] first-{T-128} K diff = {first_k_diff:.6e}  (must be > 0, quantised)")
    print(f"[R=128] first-{T-128} V diff = {first_v_diff:.6e}  (must be > 0, quantised)")
    assert last_k_diff == 0.0, "K residual not preserved"
    assert last_v_diff == 0.0, "V residual not preserved"
    assert first_k_diff > 0.0, "K prefix not quantised"
    assert first_v_diff > 0.0, "V prefix not quantised"

    # ── Case 2: R=0, expect both fully quantised (legacy path) ─────────────────
    m = make_module(R=0, kbits=2, vbits=2)
    Kq2, Vq2 = quant_kv_output(m, K.clone(), V.clone(), attention_mask=None)
    full_k_diff = (Kq2 - K).abs().max().item()
    full_v_diff = (Vq2 - V).abs().max().item()
    print(f"[R=0]  full K diff = {full_k_diff:.6e}  (must be > 0)")
    print(f"[R=0]  full V diff = {full_v_diff:.6e}  (must be > 0)")
    assert full_k_diff > 0.0 and full_v_diff > 0.0

    # ── Case 3: T <= R, expect everything FP ───────────────────────────────────
    m = make_module(R=128, kbits=2, vbits=2)
    K_short = torch.randn(B, H, 64, D, dtype=torch.float16, device="cuda")
    V_short = torch.randn(B, H, 64, D, dtype=torch.float16, device="cuda")
    Kq3, Vq3 = quant_kv_output(m, K_short.clone(), V_short.clone(), attention_mask=None)
    print(f"[T<=R] short K identical: {(Kq3 == K_short).all().item()}")
    print(f"[T<=R] short V identical: {(Vq3 == V_short).all().item()}")
    assert (Kq3 == K_short).all() and (Vq3 == V_short).all()

    print("\nAll quant_kv_output residual-aware checks PASSED")


if __name__ == "__main__":
    main()
