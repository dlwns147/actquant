"""Analyze cross-SEQLEN comparability for the expanded arch pool.

Reads data/random_arch_jsd.json (9 fixed + N random archs).
Produces:
  figures/random_arch_pareto.png       — Pareto front (avg_bits, JSD) per SEQLEN
  figures/random_arch_corr.png         — Pairwise scatter + R² + Spearman matrix
  figures/random_arch_rank_consistency.png — rank shift across SEQLENs
"""
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "data" / "random_arch_jsd.json"))

seqlens = sorted({int(s) for r in data for s in r["jsd"].keys()})
N = len(data)
fixed_idx = [i for i, r in enumerate(data) if r["type"] == "fixed"]
rand_idx = [i for i, r in enumerate(data) if r["type"] == "random"]

bits = np.array([r["avg_bits"] for r in data])
gen = {s: np.array([float(np.mean(r["jsd"][str(s)])) for r in data]) for s in seqlens}

from scipy.stats import spearmanr, kendalltau, pearsonr

# ── Tables ────────────────────────────────────────────────────────────
print(f"Archs: total={N}  fixed={len(fixed_idx)}  random={len(rand_idx)}")
print(f"Seqlens: {seqlens}")
print(f"avg_bits range: [{bits.min():.3f}, {bits.max():.3f}]")

print("\n=== Pairwise cross-SEQLEN correlation over expanded arch pool ===")
print(f"{'pair':>14}  {'Pearson':>8}  {'R^2':>6}  {'Spearman':>9}  {'Kendall':>8}")
for i, sa in enumerate(seqlens):
    for sb in seqlens[i+1:]:
        rp, _ = pearsonr(gen[sa], gen[sb])
        rs = spearmanr(gen[sa], gen[sb]).correlation
        rk = kendalltau(gen[sa], gen[sb]).correlation
        print(f"{sa:>5}↔{sb:<5}  {rp:+8.4f}  {rp**2:6.4f}  {rs:+9.4f}  {rk:+8.4f}")

# Pareto: minimise both bits and JSD
def pareto_idx(x, y):
    order = np.argsort(x)  # ascending bits
    pf, best = [], np.inf
    for i in order:
        if y[i] < best:
            pf.append(i); best = y[i]
    return pf

pareto = {s: pareto_idx(bits, gen[s]) for s in seqlens}
print("\n=== Pareto-optimal archs per seqlen ===")
for s in seqlens:
    print(f"  seq={s}: {[data[i]['name'] for i in pareto[s]]}")

print("\n=== Pareto Jaccard overlap ===")
for i, sa in enumerate(seqlens):
    for sb in seqlens[i+1:]:
        A = set(data[i]["name"] for i in pareto[sa])
        B = set(data[j]["name"] for j in pareto[sb])
        j = len(A & B) / len(A | B) if A | B else 1.0
        print(f"  {sa}↔{sb}: |A|={len(A)} |B|={len(B)} |A∩B|={len(A&B)} Jaccard={j:.3f}")

# How often are random archs dominated by fixed-precision configs?
print("\n=== Fraction of random archs dominated by ANY fixed config (per SEQLEN) ===")
for s in seqlens:
    dom = 0
    for i in rand_idx:
        bx, jx = bits[i], gen[s][i]
        for f in fixed_idx:
            if bits[f] <= bx and gen[s][f] <= jx and (bits[f] < bx or gen[s][f] < jx):
                dom += 1; break
    print(f"  seq={s}: {dom}/{len(rand_idx)} random archs dominated")

# ── Figure 1: Pareto front overlay ────────────────────────────────────
fig, axes = plt.subplots(1, len(seqlens), figsize=(7 * len(seqlens), 6), sharey=False)
for ax, s in zip(axes, seqlens):
    ax.scatter(bits[fixed_idx], gen[s][fixed_idx], s=120, marker="s",
               color="#cc3333", edgecolor="black", lw=0.6, label="fixed", zorder=3)
    ax.scatter(bits[rand_idx], gen[s][rand_idx], s=70, marker="o",
               color="#444", edgecolor="black", lw=0.4, alpha=0.7, label="random",
               zorder=2)
    pf = pareto[s]
    ax.plot(bits[pf], gen[s][pf], "-", color="#1f77b4", lw=2,
            label="Pareto", zorder=4)
    for i in pf:
        ax.annotate(data[i]["name"], (bits[i], gen[s][i]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                    color="#1f77b4")
    # annotate fixed configs
    for i in fixed_idx:
        ax.annotate(data[i]["name"], (bits[i], gen[s][i]),
                    fontsize=6, xytext=(4, -8), textcoords="offset points",
                    color="#cc3333", alpha=0.8)
    ax.set_xlabel("avg KV bits  (= mean(K + V) / 2 over layers)")
    ax.set_ylabel("L_gen JSD")
    ax.set_title(f"seq = {s}")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
fig.suptitle(f"Pareto front per SEQLEN: {len(fixed_idx)} fixed + {len(rand_idx)} random archs", y=1.02)
plt.tight_layout()
out = HERE / "figures" / "random_arch_pareto.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")

# ── Figure 2: Cross-SEQLEN scatter (3 pair plots) ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
pairs = [(seqlens[i], seqlens[j]) for i in range(len(seqlens)) for j in range(i+1, len(seqlens))]
for ax, (sa, sb) in zip(axes, pairs):
    ax.scatter(gen[sa][fixed_idx], gen[sb][fixed_idx], s=110, marker="s",
               color="#cc3333", edgecolor="black", label="fixed", zorder=3)
    ax.scatter(gen[sa][rand_idx], gen[sb][rand_idx], s=60, marker="o",
               color="#444", edgecolor="black", lw=0.3, alpha=0.7, label="random")
    lo = min(gen[sa].min(), gen[sb].min()) * 0.9
    hi = max(gen[sa].max(), gen[sb].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.4, label="y=x")
    rp, _ = pearsonr(gen[sa], gen[sb])
    rs = spearmanr(gen[sa], gen[sb]).correlation
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"L_gen JSD @ seq={sa}")
    ax.set_ylabel(f"L_gen JSD @ seq={sb}")
    ax.set_title(f"{sa} vs {sb}\nR²={rp**2:.4f}  Spearman={rs:.4f}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
plt.tight_layout()
out2 = HERE / "figures" / "random_arch_corr.png"
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out2}")

# ── Figure 3: Rank consistency (rank under each SEQLEN) ───────────────
fig, ax = plt.subplots(figsize=(max(10, N * 0.3), 6))
ranks = {s: np.argsort(np.argsort(gen[s])) for s in seqlens}
x = np.arange(N)
seq_color = {s: c for s, c in zip(seqlens, ["#1f77b4", "#ff7f0e", "#2ca02c"])}
w = 0.27
for i, s in enumerate(seqlens):
    ax.bar(x + (i - 1) * w, ranks[s] + 1, w, color=seq_color[s],
           edgecolor="black", lw=0.3, label=f"seq={s}")
order = np.argsort(gen[seqlens[0]])
ax.set_xticks(x)
ax.set_xticklabels([data[i]["name"] for i in order], rotation=80, fontsize=7)
# Re-index y values to match xtick order
for i, s in enumerate(seqlens):
    pass  # already plotted in original order
# Actually re-plot in sorted order for readability
ax.cla()
for i, s in enumerate(seqlens):
    ax.bar(x + (i - 1) * w, ranks[s][order] + 1, w, color=seq_color[s],
           edgecolor="black", lw=0.3, label=f"seq={s}")
ax.set_xticks(x)
ax.set_xticklabels([data[idx]["name"] for idx in order], rotation=80, fontsize=7)
ax.set_ylabel("rank by L_gen JSD  (1 = lowest)")
ax.set_title(f"Per-arch rank across SEQLENs ({N} archs, sorted by seq=2k rank)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
out3 = HERE / "figures" / "random_arch_rank_consistency.png"
plt.savefig(out3, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out3}")
