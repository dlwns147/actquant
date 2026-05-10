"""
KVBITS-axis answer-token / stride sweep analysis.

For COMP_OBJ=kvbits, the search optimizes per-layer KV bit-width in {2,3,4}
with group-size in {32,64,128}. The "right end" (most expensive, highest
fidelity) is kvbits=4 across all layers; the "left end" is kvbits=2.

We measure the same diagnostic metrics as the kvdim sweep:
  - PF max kvbits (does the search reach the right edge ~4?)
  - PF loss spread (signal magnitude)
  - Within-bin metric std (noise floor)
  - SNR = spread / within-bin std

Run dirs are picked by glob pattern (newest matching).
"""
import os
import json
import glob
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

THINK_DIR = "/NAS/SJ/actquant/search/save/search/think"
OUT_DIR = "/NAS/SJ/actquant/search/analysis/answer_stride"
FIG_DIR = os.path.join(OUT_DIR, "figures_kvbits")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# (label, dir-glob, ans, stride, n_sample, total_seq)
RUNS = [
    ("kvb_a128_s128",   "*kvbits_loss*kivi_iter_150*1bs_32sample_1920seq_*128stride_pp128",  128, 128, 32, 2048),
    ("kvb_a512_s128",   "*kvbits_loss*kivi_iter_150*1bs_32sample_1536seq_*128stride_pp512",  512, 128, 32, 2048),
    ("kvb_a1024_s128",  "*kvbits_loss*kivi_iter_150*1bs_32sample_2048seq_*128stride_pp1024",1024, 128, 32, 3072),
    ("kvb_a1024_s512",  "*kvbits_loss*kivi_iter_150*1bs_32sample_2048seq_*512stride_pp1024",1024, 512, 32, 3072),
]


def find_run_dir(pattern):
    matches = sorted(glob.glob(os.path.join(THINK_DIR, pattern)))
    return matches[-1] if matches else None


def kvbits_per_layer(arch):
    """Returns per-layer K bits (q['k'][l][0])."""
    k = arch["q"]["k"]
    return np.array([entry[0] for entry in k], dtype=np.float32)


def pareto_front_2d(metric, comp):
    n = len(metric)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((comp, metric))
    keep = np.zeros(n, dtype=bool)
    best = np.inf
    last_metric = -np.inf
    for k in order:
        if comp[k] < best:
            keep[k] = True
            best = comp[k]
            last_metric = metric[k]
    return keep


def list_iters(d):
    files = glob.glob(os.path.join(d, "iter_*.stats"))
    nums = []
    for f in files:
        try:
            nums.append(int(os.path.basename(f).replace("iter_", "").replace(".stats", "")))
        except ValueError:
            pass
    return sorted(nums)


# ───────── analyze each run ─────────
summary = []
series = {}

for label, pat, ans, stride, nsamp, total in RUNS:
    d = find_run_dir(pat)
    if d is None:
        print(f"[skip] {label}: no dir matching {pat}")
        continue
    iters = list_iters(d)
    if not iters:
        print(f"[skip] {label}: no iter stats yet")
        continue
    print(f"[load] {label}: {len(iters)} iters from {os.path.basename(d)}")

    series[label] = {
        "ans": ans, "stride": stride, "nsamp": nsamp, "total": total,
        "iters": iters,
        "pf_max_kvbits": [], "pf_min_kvbits": [],
        "pf_size": [], "pf_loss_spread": [],
        "mean_kvbits": [], "frac_kvbits4": [],
    }

    for it in iters:
        with open(os.path.join(d, f"iter_{it}.stats")) as f:
            st = json.load(f)
        archive = st["archive"]
        metric = np.array([x[1] for x in archive])
        comp = np.array([x[2] for x in archive])
        layer_kvbits = np.stack([kvbits_per_layer(x[0]) for x in archive])

        pf = pareto_front_2d(metric, comp)
        series[label]["pf_max_kvbits"].append(float(comp[pf].max()))
        series[label]["pf_min_kvbits"].append(float(comp[pf].min()))
        series[label]["pf_size"].append(int(pf.sum()))
        series[label]["pf_loss_spread"].append(
            float(metric[pf].max() - metric[pf].min()))
        series[label]["mean_kvbits"].append(float(layer_kvbits.mean()))
        series[label]["frac_kvbits4"].append(float((layer_kvbits == 4).mean()))

    last = iters[-1]
    with open(os.path.join(d, f"iter_{last}.stats")) as f:
        st = json.load(f)
    metric = np.array([x[1] for x in st["archive"]])
    comp = np.array([x[2] for x in st["archive"]])
    series[label]["last_metric"] = metric
    series[label]["last_comp"] = comp

    # within-bin std (noise floor proxy)
    bins = np.linspace(comp.min(), comp.max() + 1e-6, 9)
    inds = np.digitize(comp, bins) - 1
    within = []
    for b in range(8):
        m = metric[inds == b]
        if len(m) >= 5:
            within.append(m.std())
    within_std = float(np.mean(within)) if within else float("nan")

    summary.append({
        "label": label,
        "ans": ans,
        "stride": stride,
        "n_sample": nsamp,
        "total_seq": total,
        "n_iters": len(iters),
        "final_iter": last,
        "pf_max_kvbits": series[label]["pf_max_kvbits"][-1],
        "pf_min_kvbits": series[label]["pf_min_kvbits"][-1],
        "pf_size": series[label]["pf_size"][-1],
        "pf_loss_spread": series[label]["pf_loss_spread"][-1],
        "mean_kvbits": series[label]["mean_kvbits"][-1],
        "frac_kvbits4": series[label]["frac_kvbits4"][-1],
        "within_bin_std": within_std,
        "snr": (series[label]["pf_loss_spread"][-1] / within_std)
               if within_std > 0 else float("nan"),
    })

# ── csv ──
csv_path = os.path.join(DATA_DIR, "summary_kvbits.csv")
if summary:
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"[write] {csv_path}")

# ── plots ──
COLORS = plt.get_cmap("tab10").colors


def fig_curve(field, ylabel, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, s) in enumerate(series.items()):
        ax.plot(s["iters"], s[field], "-o", ms=3,
                label=f"a={s['ans']},s={s['stride']}",
                color=COLORS[i % len(COLORS)])
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[write] {out}")


if series:
    fig_curve("pf_max_kvbits",
              "Pareto front max kvbits (right-end coverage; max=4)",
              "pareto_max_kvbits.png", ylim=(2, 4.2))
    fig_curve("pf_loss_spread", "Pareto-front loss spread", "pf_loss_spread.png")
    fig_curve("mean_kvbits", "archive mean kvbits", "mean_kvbits.png",
              ylim=(2, 4.2))
    fig_curve("frac_kvbits4", "fraction of layers at kvbits=4 in archive",
              "frac_kvbits4.png", ylim=(0, 1))

    # final pareto scatter
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for i, (label, s) in enumerate(series.items()):
        m = s["last_metric"]
        c = s["last_comp"]
        pf = pareto_front_2d(m, c)
        ax = axes[i]
        ax.scatter(c, m, s=6, c="lightgray", label="archive")
        ax.scatter(c[pf], m[pf], s=20, c="red",
                   edgecolors="k", linewidths=0.4, label="Pareto")
        ax.axvline(4, color="green", lw=0.5, ls="--", label="kvbits=4")
        ax.set_xlim(c.min() - 0.1, 4.2)
        ax.set_xlabel("kvbits (avg)")
        ax.set_title(f"a={s['ans']},s={s['stride']}\niter={s['iters'][-1]}",
                     fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss (JSD)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "final_pareto_scatter.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[write] {out}")

# ── report ──
print()
print("=" * 110)
print("KVBITS exploration summary (final iteration)")
print("=" * 110)
hdr = (f"{'label':<18s} {'ans':>4s} {'stride':>6s} {'#iter':>5s} "
       f"{'mean_kvb':>9s} {'%kvb=4':>7s} "
       f"{'pf_max':>6s} {'pf_min':>6s} {'pf_n':>4s} "
       f"{'pf_dL':>8s} {'wb_std':>8s} {'SNR':>5s}")
print(hdr)
print("-" * len(hdr))
for r in summary:
    print(f"{r['label']:<18s} {r['ans']:>4d} {r['stride']:>6d} "
          f"{r['n_iters']:>5d} {r['mean_kvbits']:>9.3f} "
          f"{r['frac_kvbits4']:>7.3f} "
          f"{r['pf_max_kvbits']:>6.2f} {r['pf_min_kvbits']:>6.2f} "
          f"{r['pf_size']:>4d} {r['pf_loss_spread']:>8.4f} "
          f"{r['within_bin_std']:>8.5f} "
          f"{r['snr']:>5.1f}")
