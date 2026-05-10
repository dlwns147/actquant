"""Extended cross-SEQLEN comparability over fixed + mixed-precision random pool.

Reads data/random_arch_jsd.json (9 fixed + N random archs × 3 seqlens × n_samples).

Goes beyond 10_plot_random_arch.py by stratifying the analysis along axes that
matter for search:
  1) Stratified correlation: combined / fixed-only / random-only / within-bit-bucket.
     The combined Pearson is inflated by the bit-axis — it does not tell us
     whether *ranking among archs of similar bit-budget* survives a seqlen change.
  2) Top-K rank stability: top-3 / top-5 / top-10 overlap (Jaccard).
  3) Pareto carry-over: for each pair (sa, sb), what happens to sa's Pareto
     members at sb — still Pareto, dominated by, or just non-dominated.
  4) Rank-shift distribution split by arch type (fixed vs random).
  5) Cross-seq linear fit (slope / intercept) — does JSD scale linearly?

Output: figures/seqlen_extended.png, figures/seqlen_within_bucket.png + tables.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
data = json.load(open(HERE / "data" / "random_arch_jsd.json"))

seqlens = sorted({int(s) for r in data for s in r["jsd"].keys()})
N = len(data)
fixed_idx = np.array([i for i, r in enumerate(data) if r["type"] == "fixed"])
rand_idx = np.array([i for i, r in enumerate(data) if r["type"] == "random"])

bits = np.array([r["avg_bits"] for r in data])
gen = {s: np.array([float(np.mean(r["jsd"][str(s)])) for r in data]) for s in seqlens}
names = [r["name"] for r in data]

print(f"Archs: total={N}  fixed={len(fixed_idx)}  random={len(rand_idx)}")
print(f"Seqlens: {seqlens}   avg_bits range [{bits.min():.2f}, {bits.max():.2f}]")

# ─────────── 1) Stratified correlations ──────────────────────────────
def corr(a, b):
    rp = pearsonr(a, b)[0]
    rs = spearmanr(a, b).correlation
    rk = kendalltau(a, b).correlation
    return rp, rp**2, rs, rk

print("\n=== Stratified Pearson / R^2 / Spearman / Kendall ===")
print(f"{'pair':>11}  {'split':>14}  {'n':>3}  {'Pearson':>8}  {'R²':>6}  {'Spearman':>9}  {'Kendall':>8}")
splits = [("combined", np.arange(N)),
          ("fixed-only", fixed_idx),
          ("random-only", rand_idx)]
# Within-bit-bucket: bin by avg_bits in 0.5 increments
buckets = []
for lo, hi in [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0001)]:
    sel = np.where((bits >= lo) & (bits < hi))[0]
    if len(sel) >= 4:
        buckets.append((f"bits∈[{lo:.1f},{hi:.1f})", sel))
splits += buckets

stratified = {}
for sa, sb in [(seqlens[i], seqlens[j]) for i in range(len(seqlens)) for j in range(i+1, len(seqlens))]:
    for label, idx in splits:
        if len(idx) < 3:
            continue
        rp, r2, rs, rk = corr(gen[sa][idx], gen[sb][idx])
        stratified[(sa, sb, label)] = (rp, r2, rs, rk, len(idx))
        print(f"{sa:>5}↔{sb:<5}  {label:>14}  {len(idx):>3}  "
              f"{rp:+8.4f}  {r2:6.4f}  {rs:+9.4f}  {rk:+8.4f}")

# ─────────── 2) Top-K rank stability ─────────────────────────────────
print("\n=== Top-K (lowest JSD) overlap across seqlens (Jaccard) ===")
print(f"{'pair':>11}  " + "  ".join(f"top{k}".rjust(6) for k in [3, 5, 10]))
topk_overlap = {}
for sa, sb in [(seqlens[i], seqlens[j]) for i in range(len(seqlens)) for j in range(i+1, len(seqlens))]:
    line = f"{sa:>5}↔{sb:<5}  "
    for K in [3, 5, 10]:
        A = set(np.argsort(gen[sa])[:K])
        B = set(np.argsort(gen[sb])[:K])
        j = len(A & B) / len(A | B)
        topk_overlap[(sa, sb, K)] = j
        line += f"{j:6.3f}  "
    print(line)

# ─────────── 3) Pareto carry-over (status at the other seqlen) ────────
def pareto_idx(x, y):
    order = np.argsort(x)
    out, best = [], np.inf
    for i in order:
        if y[i] < best:
            out.append(i); best = y[i]
    return out

pareto = {s: pareto_idx(bits, gen[s]) for s in seqlens}
print("\n=== Pareto carry-over: status of sa's Pareto archs evaluated at sb ===")
print(f"{'pair':>11}  {'|PF(sa)|':>9}  {'still on PF(sb)':>16}  "
      f"{'dominated@sb':>14}  {'free@sb':>9}")
for sa, sb in [(seqlens[i], seqlens[j]) for i in range(len(seqlens)) for j in range(i+1, len(seqlens))]:
    A = pareto[sa]
    Bset = set(pareto[sb])
    still = sum(1 for i in A if i in Bset)
    # An sa-PF arch is "dominated" at sb if ANY other arch beats it at (bits, gen[sb])
    dominated = 0
    for i in A:
        if any(bits[k] <= bits[i] and gen[sb][k] <= gen[sb][i]
               and (bits[k] < bits[i] or gen[sb][k] < gen[sb][i])
               for k in range(N) if k != i):
            dominated += 1
    free = len(A) - dominated  # = still + (was-dominated→non-dominant non-PF) — but in 2D that's only PF
    print(f"{sa:>5}↔{sb:<5}  {len(A):>9}  {still:>16}  {dominated:>14}  {free:>9}")

# ─────────── 4) Rank-shift distribution by type ──────────────────────
print("\n=== Mean |rank-shift| across seqlen pairs (smaller = more stable) ===")
ranks = {s: np.argsort(np.argsort(gen[s])) for s in seqlens}
print(f"{'pair':>11}  {'all':>6}  {'fixed':>6}  {'random':>6}  "
      f"{'all-max':>7}  {'rand-max':>8}")
shift_by_type = {}
for sa, sb in [(seqlens[i], seqlens[j]) for i in range(len(seqlens)) for j in range(i+1, len(seqlens))]:
    d = np.abs(ranks[sa] - ranks[sb])
    shift_by_type[(sa, sb)] = d
    print(f"{sa:>5}↔{sb:<5}  {d.mean():6.2f}  "
          f"{d[fixed_idx].mean():6.2f}  {d[rand_idx].mean():6.2f}  "
          f"{d.max():>7d}  {d[rand_idx].max():>8d}")

# ─────────── 5) Cross-seq linear fit ─────────────────────────────────
print("\n=== Linear fit  gen[sb] = a + b * gen[sa]   (b≠1 ⇒ scale shift) ===")
print(f"{'pair':>11}  {'a':>12}  {'b':>8}  {'residual_std':>13}")
fits = {}
for sa, sb in [(seqlens[i], seqlens[j]) for i in range(len(seqlens)) for j in range(i+1, len(seqlens))]:
    b_, a_ = np.polyfit(gen[sa], gen[sb], 1)
    resid = gen[sb] - (a_ + b_ * gen[sa])
    fits[(sa, sb)] = (a_, b_, resid.std())
    print(f"{sa:>5}↔{sb:<5}  {a_:>12.6f}  {b_:>8.4f}  {resid.std():>13.6f}")

# ═════════ Plot 1: extended cross-seq summary ═════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

pairs = [(seqlens[i], seqlens[j]) for i in range(len(seqlens))
         for j in range(i+1, len(seqlens))]

# Row 1: stratified Spearman bars (combined / fixed / random) per pair
ax = axes[0, 0]
labels = ["combined", "fixed-only", "random-only"]
x = np.arange(len(pairs))
w = 0.27
colors = {"combined": "#444", "fixed-only": "#cc3333", "random-only": "#1f77b4"}
for k, lab in enumerate(labels):
    vals = [stratified[(sa, sb, lab)][2] for sa, sb in pairs]
    ax.bar(x + (k - 1) * w, vals, w, color=colors[lab], edgecolor="black",
           label=lab)
ax.set_xticks(x)
ax.set_xticklabels([f"{a}↔{b}" for a, b in pairs])
ax.set_ylabel("Spearman ρ")
ax.set_title("Stratified ranking stability\n(combined inflated by bit-axis)")
ax.set_ylim(0, 1.02)
ax.axhline(1.0, color="gray", lw=0.5, ls="--")
ax.legend()
ax.grid(alpha=0.3, axis="y")

# Row 1: within-bit-bucket Spearman
ax = axes[0, 1]
bucket_labels = [b[0] for b in buckets]
ns = [len(b[1]) for b in buckets]
xb = np.arange(len(buckets))
wb = 0.27
pair_color = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for k, (sa, sb) in enumerate(pairs):
    vals = [stratified[(sa, sb, lab)][2] for lab in bucket_labels]
    ax.bar(xb + (k - 1) * wb, vals, wb, color=pair_color[k],
           edgecolor="black", label=f"{sa}↔{sb}")
ax.set_xticks(xb)
ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(bucket_labels, ns)],
                   fontsize=8)
ax.set_ylabel("Spearman ρ within bit bucket")
ax.set_title("Within-bit-bucket ranking stability\n(controls for bit-axis dominance)")
ax.axhline(0, color="gray", lw=0.5)
ax.axhline(1.0, color="gray", lw=0.5, ls="--")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# Row 1: top-K Jaccard
ax = axes[0, 2]
Ks = [3, 5, 10]
xk = np.arange(len(Ks))
wk = 0.27
for k, (sa, sb) in enumerate(pairs):
    vals = [topk_overlap[(sa, sb, K)] for K in Ks]
    ax.bar(xk + (k - 1) * wk, vals, wk, color=pair_color[k],
           edgecolor="black", label=f"{sa}↔{sb}")
ax.set_xticks(xk); ax.set_xticklabels([f"top-{K}" for K in Ks])
ax.set_ylabel("Jaccard overlap")
ax.set_title("Top-K (lowest JSD) overlap\n(what real search would pick)")
ax.set_ylim(0, 1.02)
ax.legend(); ax.grid(alpha=0.3, axis="y")

# Row 2: rank-shift hist split by type — one panel per pair
for k, (sa, sb) in enumerate(pairs):
    ax = axes[1, k]
    d = shift_by_type[(sa, sb)]
    bins = np.arange(0, max(d.max() + 2, 6))
    ax.hist(d[fixed_idx], bins=bins, alpha=0.7, color="#cc3333",
            edgecolor="black", label=f"fixed (n={len(fixed_idx)}, μ={d[fixed_idx].mean():.1f})")
    ax.hist(d[rand_idx], bins=bins, alpha=0.55, color="#1f77b4",
            edgecolor="black", label=f"random (n={len(rand_idx)}, μ={d[rand_idx].mean():.1f})")
    ax.set_xlabel("|rank shift|")
    ax.set_ylabel("# archs")
    ax.set_title(f"{sa} → {sb}  (max shift = {d.max()})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

fig.suptitle(f"Extended cross-SEQLEN analysis  ({len(fixed_idx)} fixed + {len(rand_idx)} random archs)",
             fontsize=13, y=1.00)
plt.tight_layout()
out = FIG / "seqlen_extended.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")

# ═════════ Plot 2: Pareto carry-over diagram ═══════════════════════════
fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
for ax, (sa, sb) in zip(axes, pairs):
    pf_a = set(pareto[sa])
    pf_b = set(pareto[sb])
    # Draw all archs at sb, color by membership in PF(sa)
    only_b = [i for i in range(N) if i in pf_b and i not in pf_a]
    both = [i for i in range(N) if i in pf_a and i in pf_b]
    only_a = [i for i in range(N) if i in pf_a and i not in pf_b]
    other = [i for i in range(N) if i not in pf_a and i not in pf_b]

    ax.scatter(bits[other], gen[sb][other], s=40, color="lightgray",
               edgecolor="black", lw=0.3, label=f"non-Pareto at either ({len(other)})",
               zorder=1)
    ax.scatter(bits[only_a], gen[sb][only_a], s=110, marker="x",
               color="#cc3333", lw=2,
               label=f"PF(sa)\\PF(sb): lost ({len(only_a)})", zorder=3)
    ax.scatter(bits[only_b], gen[sb][only_b], s=110, marker="^",
               color="#2ca02c", edgecolor="black",
               label=f"PF(sb)\\PF(sa): new ({len(only_b)})", zorder=3)
    ax.scatter(bits[both], gen[sb][both], s=130, marker="o",
               color="#1f77b4", edgecolor="black", lw=0.6,
               label=f"PF in both ({len(both)})", zorder=4)
    pf = pareto[sb]
    ax.plot(bits[pf], gen[sb][pf], "-", color="#1f77b4", lw=1.5, alpha=0.4)
    ax.set_xlabel("avg KV bits")
    ax.set_ylabel(f"L_gen JSD @ seq={sb}")
    ax.set_title(f"PF carry-over: PF(seq={sa}) → seq={sb}\n"
                 f"|both|/|sa|={len(both)}/{len(pf_a)} = {len(both)/len(pf_a):.2f}")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
plt.tight_layout()
out2 = FIG / "seqlen_pareto_carryover.png"
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out2}")

# ═════════ Plot 3: cross-seq scatter w/ linear fit + residuals ═════════
fig, axes = plt.subplots(2, len(pairs), figsize=(7 * len(pairs), 11))
for k, (sa, sb) in enumerate(pairs):
    a_, b_, sd = fits[(sa, sb)]
    # Top: scatter with fit
    ax = axes[0, k]
    ax.scatter(gen[sa][fixed_idx], gen[sb][fixed_idx], s=110, marker="s",
               color="#cc3333", edgecolor="black", label="fixed", zorder=3)
    ax.scatter(gen[sa][rand_idx], gen[sb][rand_idx], s=55, marker="o",
               color="#1f77b4", edgecolor="black", lw=0.3, alpha=0.75,
               label="random", zorder=2)
    xs = np.linspace(gen[sa].min(), gen[sa].max(), 100)
    ax.plot(xs, a_ + b_ * xs, "-", color="black", lw=1.4,
            label=f"fit: {b_:.3f}·x{a_:+.4f}")
    lo = min(gen[sa].min(), gen[sb].min()) * 0.9
    hi = max(gen[sa].max(), gen[sb].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.7, label="y=x")
    ax.set_xlabel(f"L_gen JSD @ seq={sa}")
    ax.set_ylabel(f"L_gen JSD @ seq={sb}")
    ax.set_title(f"{sa} → {sb}  (slope={b_:.3f}, σ_resid={sd:.4f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    # Bottom: residuals by avg_bits
    ax = axes[1, k]
    resid = gen[sb] - (a_ + b_ * gen[sa])
    ax.scatter(bits[fixed_idx], resid[fixed_idx], s=110, marker="s",
               color="#cc3333", edgecolor="black", label="fixed", zorder=3)
    ax.scatter(bits[rand_idx], resid[rand_idx], s=55, marker="o",
               color="#1f77b4", edgecolor="black", lw=0.3, alpha=0.75,
               label="random", zorder=2)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("avg KV bits")
    ax.set_ylabel(f"residual gen[{sb}] − fit")
    ax.set_title(f"residual vs bits  (μ={resid.mean():+.4f}, σ={sd:.4f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
out3 = FIG / "seqlen_linear_fit.png"
plt.savefig(out3, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out3}")
