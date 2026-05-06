"""
Generate N candidate quant configs for Llama-3.1-8B-Instruct spanning
W and KV bit variation. Output: candidates.json — list of arch dicts
in the schema consumed by LlamaEvaluator.sample(arch).
"""
import argparse
import json
import os
import random


LINEARS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def uniform_w(n_block, w_bit):
    return {ln: [w_bit] * n_block for ln in LINEARS}


def uniform_kv(n_block, kv_bit, gs=128):
    return [[kv_bit, gs] for _ in range(n_block)]


def random_w(n_block, w_bits, rng):
    return {ln: [rng.choice(w_bits) for _ in range(n_block)] for ln in LINEARS}


def random_kv(n_block, kv_bits, gs, rng):
    return [[rng.choice(kv_bits), gs] for _ in range(n_block)]


def make_arch(w, k, v, p_k=None, p_v=None):
    arch = {"q": {"w": w, "k": k, "v": v}}
    if p_k is not None or p_v is not None:
        arch["p"] = {}
        if p_k is not None:
            arch["p"]["k"] = p_k
        if p_v is not None:
            arch["p"]["v"] = p_v
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_block", type=int, default=32)
    ap.add_argument("--w_bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--kv_bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--kv_gs", type=int, default=128)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arches = []

    # Uniform W × KV grid (3×3 = 9 corner configs).
    for wb in args.w_bits:
        for kvb in args.kv_bits:
            w = uniform_w(args.n_block, wb)
            k = uniform_kv(args.n_block, kvb, args.kv_gs)
            v = uniform_kv(args.n_block, kvb, args.kv_gs)
            arches.append(make_arch(w, k, v))

    # KV-only sweep at W=4 with mixed KV per layer (K & V independent).
    for _ in range(6):
        w = uniform_w(args.n_block, 4)
        k = random_kv(args.n_block, args.kv_bits, args.kv_gs, rng)
        v = random_kv(args.n_block, args.kv_bits, args.kv_gs, rng)
        arches.append(make_arch(w, k, v))

    # W-only sweep at KV=4 with mixed W per linear/layer.
    for _ in range(6):
        w = random_w(args.n_block, args.w_bits, rng)
        k = uniform_kv(args.n_block, 4, args.kv_gs)
        v = uniform_kv(args.n_block, 4, args.kv_gs)
        arches.append(make_arch(w, k, v))

    # Mixed: random W and random KV.
    while len(arches) < args.n:
        w = random_w(args.n_block, args.w_bits, rng)
        k = random_kv(args.n_block, args.kv_bits, args.kv_gs, rng)
        v = random_kv(args.n_block, args.kv_bits, args.kv_gs, rng)
        arches.append(make_arch(w, k, v))

    arches = arches[: args.n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(arches, f, indent=2)
    print(f"Wrote {len(arches)} candidates to {args.out}")


if __name__ == "__main__":
    main()
