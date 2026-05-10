"""
Corrected kvbits analysis: the right end of comp_obj=kvbits is **5.0** bits
(per-layer 4-bit + gs=32 → 4 + 32/32 = 5), NOT 4.0. Left end is 2.25
(2-bit + gs=128 → 2 + 0.25 = 2.25). True comp range is [2.25, 5.0], width 2.75.

We re-evaluate right-end coverage with the correct normalization, and break
down per-layer (bits, group_size) distribution at the Pareto front to see
whether the search reaches the (4bit, gs=32) corner.
"""
import os
import json
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

THINK_DIR = "/NAS/SJ/actquant/search/save/search/think"
DAT = "/NAS/SJ/actquant/search/analysis/answer_stride/data"
FIG = "/NAS/SJ/actquant/search/analysis/answer_stride/figures_kvbits"
os.makedirs(FIG, exist_ok=True)

LEFT_END = 2.25      # 2-bit + gs=128
RIGHT_END = 5.0      # 4-bit + gs=32
WIDTH = RIGHT_END - LEFT_END

RUNS = [
    ("kvb_a128_s128",   "*kvbits_loss*kivi_iter_150*1bs_32sample_1920seq_*128stride_pp128",  128, 128),
    ("kvb_a512_s128",   "*kvbits_loss*kivi_iter_150*1bs_32sample_1536seq_*128stride_pp512",  512, 128),
    ("kvb_a1024_s128",  "*kvbits_loss*kivi_iter_150*1bs_32sample_2048seq_*128stride_pp1024",1024, 128),
    ("kvb_a1024_s512",  "*kvbits_loss*kivi_iter_150*1bs_32sample_2048seq_*512stride_pp1024",1024, 512),
]


def find_dir(pat):
    m = sorted(glob.glob(os.path.join(THINK_DIR, pat)))
    return m[-1] if m else None


def kv_layers_bg(arch):
    """Returns concatenated K+V layer (bits, gs) tuples."""
    return list(arch["q"]["k"]) + list(arch["q"]["v"])


def pareto_front_2d(metric, comp):
    n = len(metric)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((comp, metric))
    keep = np.zeros(n, dtype=bool)
    best = np.inf
    for k in order:
        if comp[k] < best:
            keep[k] = True
            best = comp[k]
    return keep


# ───────── per-run analysis ─────────
print("=" * 100)
print(f"COMP RANGE: left={LEFT_END} (2bit/gs=128), right={RIGHT_END} (4bit/gs=32), width={WIDTH}")
print("=" * 100)
hdr = (f"{'label':<16s} {'ans':>4} {'stride':>6} {'#iter':>5} "
       f"{'PF_max':>7} {'coverage':>8} {'PF_min':>7} {'PF_dL':>7} "
       f"{'right_arch_count':>17}")
print(hdr); print("-" * len(hdr))

summary = []
series = {}

for label, pat, ans, stride in RUNS:
    d = find_dir(pat)
    if d is None:
        print(f"{label}: skip"); continue
    iter_files = sorted(glob.glob(os.path.join(d, "iter_*.stats")),
                        key=lambda f: int(f.split("iter_")[-1].split(".")[0]))
    if not iter_files:
        print(f"{label}: no iters"); continue

    series[label] = {"iters": [], "pf_max": [], "pf_cov": [], "pf_dL": [],
                     "frac_high": []}  # frac of layers with eff_bits>=4.5
    for f in iter_files:
        it = int(f.split("iter_")[-1].split(".")[0])
        with open(f) as fh:
            st = json.load(fh)
        archive = st["archive"]
        m = np.array([x[1] for x in archive])
        c = np.array([x[2] for x in archive])
        pf = pareto_front_2d(m, c)
        pf_max = float(c[pf].max())
        cov = (pf_max - LEFT_END) / WIDTH
        # per-layer effective bits across archive
        per_layer_eff = []
        for x in archive:
            for (bits, gs) in kv_layers_bg(x[0]):
                per_layer_eff.append(bits + 32 / gs if gs > 0 else bits)
        per_layer_eff = np.array(per_layer_eff)
        frac_high = float((per_layer_eff >= 4.5).mean())  # 4-bit gs=64 or smaller
        series[label]["iters"].append(it)
        series[label]["pf_max"].append(pf_max)
        series[label]["pf_cov"].append(cov)
        series[label]["pf_dL"].append(float(m[pf].max() - m[pf].min()))
        series[label]["frac_high"].append(frac_high)

    last_f = iter_files[-1]
    with open(last_f) as fh:
        st = json.load(fh)
    archive = st["archive"]
    m = np.array([x[1] for x in archive])
    c = np.array([x[2] for x in archive])
    pf = pareto_front_2d(m, c)

    # how many archs on PF have eff_bits ≥ 4.5 in MOST layers (=close to right end)
    right_arch_count = 0
    for x in [archive[i] for i, t in enumerate(pf) if t]:
        layers = kv_layers_bg(x[0])
        eff = [b + 32 / g for (b, g) in layers]
        if np.mean(eff) >= 4.5:
            right_arch_count += 1

    pf_max = float(c[pf].max())
    pf_min = float(c[pf].min())
    cov = (pf_max - LEFT_END) / WIDTH
    pf_dL = float(m[pf].max() - m[pf].min())
    print(f"{label:<16s} {ans:>4d} {stride:>6d} {len(iter_files):>5d} "
          f"{pf_max:>7.3f} {cov*100:>7.1f}% {pf_min:>7.3f} {pf_dL:>7.4f} "
          f"{right_arch_count:>17d}")

    summary.append({
        "label": label, "ans": ans, "stride": stride, "n_iters": len(iter_files),
        "pf_max": pf_max, "coverage": cov,
        "pf_min": pf_min, "pf_dL": pf_dL,
        "right_arch_count": right_arch_count,
    })

# ───────── group_size distribution at PF right edge ─────────
print()
print("Per-layer (bits, gs) histogram at the top-30 right-most PF points:")
for label, pat, ans, stride in RUNS:
    d = find_dir(pat)
    if d is None: continue
    iter_files = sorted(glob.glob(os.path.join(d, "iter_*.stats")),
                        key=lambda f: int(f.split("iter_")[-1].split(".")[0]))
    if not iter_files: continue
    with open(iter_files[-1]) as fh:
        st = json.load(fh)
    archive = st["archive"]
    m = np.array([x[1] for x in archive])
    c = np.array([x[2] for x in archive])
    pf = pareto_front_2d(m, c)
    pf_archs = [archive[i] for i, t in enumerate(pf) if t]
    # take top-K by comp (most expensive=rightmost)
    pf_archs.sort(key=lambda a: a[2], reverse=True)
    top = pf_archs[:30]
    layers = []
    for x in top:
        layers.extend(kv_layers_bg(x[0]))
    cnt = Counter((b, g) for (b, g) in layers)
    total = sum(cnt.values())
    print(f"\n  {label} (top-30 rightmost PF, {len(layers)} layer-slots):")
    for (b, g), n in sorted(cnt.items()):
        eff = b + 32/g
        bar = "█" * int(n / total * 60)
        print(f"    bits={b} gs={g:>3} (eff={eff:.2f}): {n:>5} ({100*n/total:5.1f}%) {bar}")

# ───────── plots ─────────
fig, ax = plt.subplots(figsize=(8, 5))
COL = plt.get_cmap("tab10").colors
for i, (label, s) in enumerate(series.items()):
    ans = next(r[2] for r in RUNS if r[0] == label)
    stride = next(r[3] for r in RUNS if r[0] == label)
    ax.plot(s["iters"], s["pf_cov"], "-o", ms=4,
            label=f"a={ans}, s={stride}", color=COL[i])
ax.axhline(1.0, color="green", lw=0.5, ls="--", label="full right-end (=1)")
ax.set_xlabel("iteration")
ax.set_ylabel(f"PF max coverage  (PF_max − {LEFT_END}) / {WIDTH}")
ax.set_ylim(0, 1.1)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("kvbits: right-end coverage (corrected: max=5.0)")
fig.tight_layout()
out = os.path.join(FIG, "pareto_max_coverage_corrected.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"\n[write] {out}")

# Save corrected csv
csv_path = os.path.join(DAT, "summary_kvbits_v2.csv")
if summary:
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"[write] {csv_path}")
