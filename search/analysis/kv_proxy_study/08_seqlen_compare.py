"""Cross-seqlen comparability of L_gen JSD.

Reads data/jsd_results.json (9 configs × 3 seqlens, answer_len=128).
For each pair of seqlens computes:
  1) Spearman rank correlation across the 9 configs (architecture ranking).
  2) Pearson R between (L_gen at seq A) and (L_gen at seq B), and R^2.
  3) Pareto fronts on (avg_kv_bits, L_gen JSD) — overlap of optimal configs.

Outputs: figures/seqlen_compare.png and printed tables.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
data = json.load(open(HERE / "data" / "jsd_results.json"))

cfgs = sorted({r["cfg"] for r in data}, key=lambda c: tuple(int(x) for x in c.split(":")))
seqlens = sorted({r["prompt_len"] for r in data})

# Per-cfg average kv-bits (K and V each weighted equally — both dominate cache).
def avg_bits(cfg):
    k, v = [int(x) for x in cfg.split(":")]
    return (k + v) / 2

# Mean L_gen JSD per (cfg, seq)
gen = {seq: np.array([np.mean([r for r in data if r["cfg"] == c and r["prompt_len"] == seq][0]["gen"])
                      for c in cfgs])
       for seq in seqlens}
bits = np.array([avg_bits(c) for c in cfgs])

# ─────────────────── 1) Ranking & R² across seqlens ───────────────────
from scipy.stats import spearmanr, kendalltau, pearsonr

print("=== L_gen JSD per (cfg × seqlen) ===")
print(f"{'cfg':>5} {'bits':>5}  " + "  ".join([f"seq={s}".rjust(9) for s in seqlens]))
for i, c in enumerate(cfgs):
    line = f"{c:>5} {bits[i]:>5.1f}  "
    for s in seqlens:
        line += f"{gen[s][i]:9.5f}  "
    print(line)

print("\n=== Pairwise correlation of L_gen JSD across seqlens ===")
print(f"{'pair':>14}  {'Pearson':>8}  {'R^2':>6}  {'Spearman':>9}  {'Kendall':>8}")
for i, sa in enumerate(seqlens):
    for sb in seqlens[i+1:]:
        r_p, _ = pearsonr(gen[sa], gen[sb])
        r_s = spearmanr(gen[sa], gen[sb]).correlation
        r_k = kendalltau(gen[sa], gen[sb]).correlation
        print(f"{sa:>5}↔{sb:<5}  {r_p:+8.4f}  {r_p**2:6.4f}  {r_s:+9.4f}  {r_k:+8.4f}")

# ─────────────────── 2) Pareto front per seqlen ───────────────────────
def pareto_front(x, y):
    """Return indices of points on the Pareto front (minimise both)."""
    idx = np.argsort(x)  # sort by x ascending
    pf, best_y = [], np.inf
    for i in idx:
        if y[i] < best_y:
            pf.append(i); best_y = y[i]
    return pf

pareto = {seq: pareto_front(bits, gen[seq]) for seq in seqlens}

print("\n=== Pareto-optimal configs per seqlen (lower bits / lower JSD) ===")
for seq in seqlens:
    pf_cfgs = [cfgs[i] for i in pareto[seq]]
    print(f"  seq={seq}: {pf_cfgs}")

# Jaccard overlap of Pareto sets
print("\n=== Pareto-set Jaccard overlap ===")
for i, sa in enumerate(seqlens):
    for sb in seqlens[i+1:]:
        A = set(cfgs[i] for i in pareto[sa])
        B = set(cfgs[i] for i in pareto[sb])
        j = len(A & B) / len(A | B)
        print(f"  {sa}↔{sb}: |A|={len(A)} |B|={len(B)} |A∩B|={len(A&B)}  Jaccard={j:.3f}")

# ─────────────────── 3) Plot ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
seq_color = {s: c for s, c in zip(seqlens, ["#1f77b4", "#ff7f0e", "#2ca02c"])}

# (1) Pareto fronts overlay
ax = axes[0]
for seq in seqlens:
    ax.scatter(bits, gen[seq], color=seq_color[seq], s=70, edgecolor="black",
               lw=0.5, alpha=0.7, label=f"seq={seq}")
    pf = pareto[seq]
    ax.plot(bits[pf], gen[seq][pf], "-", color=seq_color[seq], lw=1.8)
    for i in pf:
        ax.annotate(cfgs[i], (bits[i], gen[seq][i]),
                    fontsize=7, color=seq_color[seq],
                    xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("avg KV bits")
ax.set_ylabel("L_gen JSD (per-config mean)")
ax.set_title("Pareto front per seqlen\n(lines connect Pareto-optimal configs)")
ax.legend()
ax.grid(alpha=0.3)

# (2) Cross-seq scatter — L_gen JSD pair plots with R² and Spearman
ax = axes[1]
pair = (2048, 8192)
ax.scatter(gen[pair[0]], gen[pair[1]], s=80, color="#444", edgecolor="black", lw=0.5)
for i, c in enumerate(cfgs):
    ax.annotate(c, (gen[pair[0]][i], gen[pair[1]][i]),
                fontsize=8, xytext=(4, -3), textcoords="offset points")
lo = min(gen[pair[0]].min(), gen[pair[1]].min()) * 0.9
hi = max(gen[pair[0]].max(), gen[pair[1]].max()) * 1.1
ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.4, label="y=x")
r_p, _ = pearsonr(gen[pair[0]], gen[pair[1]])
r_s = spearmanr(gen[pair[0]], gen[pair[1]]).correlation
ax.set_xlabel(f"L_gen JSD @ seq={pair[0]}")
ax.set_ylabel(f"L_gen JSD @ seq={pair[1]}")
ax.set_title(f"Cross-seq scatter ({pair[0]} vs {pair[1]})\n"
             f"R²={r_p**2:.4f}  Spearman={r_s:.4f}")
ax.legend(); ax.grid(alpha=0.3)

# (3) Rank consistency — rank of each config across seqlens
ax = axes[2]
ranks = {seq: np.argsort(np.argsort(gen[seq])) for seq in seqlens}  # 0 = lowest JSD
x = np.arange(len(cfgs))
w = 0.27
for i, seq in enumerate(seqlens):
    ax.bar(x + (i - 1) * w, ranks[seq] + 1, w, color=seq_color[seq],
           edgecolor="black", label=f"seq={seq}")
ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=30)
ax.set_ylabel("rank by L_gen JSD  (1 = lowest)")
ax.set_title("Per-config rank across seqlens")
ax.legend(); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = FIG / "seqlen_compare.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"\nSaved → {out}")
