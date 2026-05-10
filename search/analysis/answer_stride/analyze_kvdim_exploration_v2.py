"""
Analyze how the (answer_token x stride) configuration affects KVDIM exploration
behavior in search_think.py runs.

Question: KVDIM-axis search collapses toward low/mid kvdim values (16-64) and
neglects the right edge (128 = no pruning). We hypothesize this is because, for
short answer-token windows, loss differences between kvdim=128 and a moderately
pruned config are small, so the Pareto front excludes the right edge.

We sweep across the existing pp{128,256,512,1024} x stride{32,64,128,512}
runs, then per iteration compute:
  - per-block mean kvdim of the archive (signal: search-space coverage)
  - fraction of archive entries with at least one layer at kvdim=128
  - Pareto-front kvdim coverage (does the front reach 128?)
  - loss spread (max-min) over Pareto front (signal: discriminability)
  - mean "K-prune" / "V-prune" amount (i.e., 128 - remained_dim)

Output:
  - figures/kvdim_distribution.png
  - figures/pareto_max_kvdim.png
  - figures/loss_spread.png
  - data/summary.csv
"""

import os
import json
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

THINK_DIR = "/NAS/SJ/actquant/search/save/search/think"
OUT_DIR = "/NAS/SJ/actquant/search/analysis/answer_stride"
FIG_DIR = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Map run dir -> (last_tokens, stride, n_sample, total_seq)
# We focus on the recent kvdim search_think runs that vary pp & stride.
RUNS = {
    "pp128_s32":            ("2605071157", 128, 32),
    "pp256_s64":            ("2605070434", 256, 64),
    "pp256_s128":           ("2605071142", 256, 128),
    "pp512_s128_s2048":     ("2605070828", 512, 128),
    "pp512_s128":           ("2605071156", 512, 128),
    "pp512_s128_n64":       ("2605070643", 512, 128),
    "pp1024_s512":          ("2605070803", 1024, 512),
    # follow-up sweep launched 2026-05-07 19:29:
    # NOTE: pp1024_s128_v2 was DEGENERATE (--seqlen 1024 == --last_tokens 1024
    # → prefill_prompt mode disabled by check `last_tokens < total_seq_len`
    # in utils/eval.py:242 → fell through to legacy stride mode, all archs
    # returned identical metric=0.020833. Re-run as exp1b at 2605080305_*.
    # "pp1024_s128_v2_BROKEN": ("2605071929_*1024seq_*128stride_pp1024", 1024, 128),
    "pp512_s64_v2":         ("2605071929_*1536seq_*64stride_pp512",     512, 64),
    "pp256_s128_resume_v2": ("2605071929_*1792seq_*128stride_pp256",    256, 128),
    "pp1024_s128_v2_fixed": ("2605080305_*2048seq_*128stride_pp1024",  1024, 128),
}


def find_run_dir(prefix):
    """prefix may already be a glob pattern (contains *)."""
    if "*" in prefix:
        pat = os.path.join(THINK_DIR, prefix)
    else:
        pat = os.path.join(THINK_DIR, f"{prefix}_*kvdim*")
    matches = sorted(glob.glob(pat))
    return matches[0] if matches else None


def kvdim_per_layer(arch):
    """Per-block remained_dim = head_dim - max(k_prune[l], v_prune[l]).
    head_dim=128 for Llama-3.1-8B; we just use the prune value
    semantics defined in search_think (p['k']/p['v'] are # channels pruned).
    Returns numpy array of length n_block.
    """
    p = arch["p"]
    pk = np.array(p["k"], dtype=np.int32)
    pv = np.array(p["v"], dtype=np.int32)
    # kvdim = remained dim = 128 - prune_amount; common usage in
    # search_space: kvdim = head_dim - max prune across K/V (per block).
    return 128 - np.maximum(pk, pv)


def k_prune_per_layer(arch):
    return np.array(arch["p"]["k"], dtype=np.int32)


def v_prune_per_layer(arch):
    return np.array(arch["p"]["v"], dtype=np.int32)


def pareto_front_2d(metric, comp):
    """Vectorized non-dominated mask in (min metric, min comp). 2D only.
    Sort by metric asc, then by comp asc as tiebreak; sweep keeping
    monotonically decreasing comp. Strictly-better duplicates are dropped.
    """
    n = len(metric)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((comp, metric))  # primary: metric, secondary: comp
    keep = np.zeros(n, dtype=bool)
    best_comp = np.inf
    last_metric = -np.inf
    for k in order:
        if comp[k] < best_comp or (comp[k] == best_comp and metric[k] < last_metric):
            keep[k] = True
            best_comp = comp[k]
            last_metric = metric[k]
        elif comp[k] <= best_comp:
            # equal-comp duplicate at higher metric: dominated
            pass
    return keep


def load_iter(run_dir, it):
    p = os.path.join(run_dir, f"iter_{it}.stats")
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def iter_metrics(d):
    archive = d["archive"]
    metric = np.array([x[1] for x in archive], dtype=np.float64)
    comp = np.array([x[2] for x in archive], dtype=np.float64)
    archs = [x[0] for x in archive]
    layer_kvdim = np.stack([kvdim_per_layer(a) for a in archs])  # (N, n_block)
    layer_kp = np.stack([k_prune_per_layer(a) for a in archs])
    layer_vp = np.stack([v_prune_per_layer(a) for a in archs])
    return metric, comp, layer_kvdim, layer_kp, layer_vp


def list_iter_numbers(run_dir):
    files = glob.glob(os.path.join(run_dir, "iter_*.stats"))
    nums = []
    for f in files:
        name = os.path.basename(f)
        try:
            nums.append(int(name.replace("iter_", "").replace(".stats", "")))
        except ValueError:
            pass
    return sorted(nums)


# ---------- main analysis ----------
summary_rows = []  # for csv

# For figures, accumulate per-run series
series = {}

for label, (prefix, ans, stride) in RUNS.items():
    run_dir = find_run_dir(prefix)
    if run_dir is None:
        print(f"[skip] {label}: dir not found")
        continue
    iters = list_iter_numbers(run_dir)
    if not iters:
        print(f"[skip] {label}: no iter stats")
        continue
    print(f"[load] {label}: {len(iters)} iters from {os.path.basename(run_dir)}")

    series[label] = {
        "ans": ans, "stride": stride,
        "iters": iters,
        "mean_kvdim": [], "max_kvdim_p95": [],
        "frac_layer_kvdim128": [],
        "pf_max_kvdim": [], "pf_min_kvdim": [], "pf_size": [],
        "pf_loss_spread": [],
        "archive_size": [],
        # final-iteration last archive entries (n_iter most recent)
        "last_metric": None,
        "last_kvdim": None,
    }

    for it in iters:
        d = load_iter(run_dir, it)
        metric, comp, lkvdim, lkp, lvp = iter_metrics(d)

        series[label]["archive_size"].append(len(metric))
        series[label]["mean_kvdim"].append(float(lkvdim.mean()))
        series[label]["max_kvdim_p95"].append(float(np.percentile(lkvdim, 95)))
        series[label]["frac_layer_kvdim128"].append(
            float((lkvdim == 128).mean()))

        pf_mask = pareto_front_2d(metric, comp)
        if pf_mask.sum() > 0:
            pf_metric = metric[pf_mask]
            pf_comp = comp[pf_mask]
            series[label]["pf_max_kvdim"].append(float(pf_comp.max()))
            series[label]["pf_min_kvdim"].append(float(pf_comp.min()))
            series[label]["pf_size"].append(int(pf_mask.sum()))
            series[label]["pf_loss_spread"].append(
                float(pf_metric.max() - pf_metric.min()))
        else:
            series[label]["pf_max_kvdim"].append(np.nan)
            series[label]["pf_min_kvdim"].append(np.nan)
            series[label]["pf_size"].append(0)
            series[label]["pf_loss_spread"].append(np.nan)

    # final-state snapshot
    d_last = load_iter(run_dir, iters[-1])
    metric, comp, lkvdim, _, _ = iter_metrics(d_last)
    series[label]["last_metric"] = metric
    series[label]["last_kvdim"] = comp

    # csv row
    summary_rows.append({
        "label": label,
        "answer_tokens": ans,
        "stride": stride,
        "n_iters": len(iters),
        "final_iter": iters[-1],
        "final_archive_size": series[label]["archive_size"][-1],
        "final_mean_kvdim": series[label]["mean_kvdim"][-1],
        "final_p95_kvdim": series[label]["max_kvdim_p95"][-1],
        "final_frac_layer_kvdim128": series[label]["frac_layer_kvdim128"][-1],
        "final_pf_max_kvdim": series[label]["pf_max_kvdim"][-1],
        "final_pf_min_kvdim": series[label]["pf_min_kvdim"][-1],
        "final_pf_size": series[label]["pf_size"][-1],
        "final_pf_loss_spread": series[label]["pf_loss_spread"][-1],
    })

# write csv
csv_path = os.path.join(DATA_DIR, "summary.csv")
keys = list(summary_rows[0].keys()) if summary_rows else []
with open(csv_path, "w") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for row in summary_rows:
        w.writerow(row)
print(f"[write] {csv_path}")

# ---------- figures ----------
COLORS = plt.get_cmap("tab10").colors


def fig_curve(field, ylabel, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, s) in enumerate(series.items()):
        ax.plot(s["iters"], s[field], "-o", ms=3,
                label=f"ans={s['ans']}, stride={s['stride']}",
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


fig_curve("mean_kvdim", "archive mean kvdim (per-block avg)",
          "mean_kvdim.png", ylim=(60, 132))
fig_curve("frac_layer_kvdim128",
          "fraction of layers with kvdim=128 in archive",
          "frac_kvdim128.png", ylim=(0, 1))
fig_curve("pf_max_kvdim",
          "Pareto front max kvdim (right-end coverage)",
          "pareto_max_kvdim.png", ylim=(60, 132))
fig_curve("pf_loss_spread",
          "Pareto-front loss spread (max-min)",
          "pf_loss_spread.png")
fig_curve("pf_size", "Pareto-front size", "pf_size.png")

# ---------- final pareto scatter ----------
fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 4),
                         sharey=True)
if len(series) == 1:
    axes = [axes]
for i, (label, s) in enumerate(series.items()):
    metric = s["last_metric"]
    comp = s["last_kvdim"]
    pf_mask = pareto_front_2d(metric, comp)
    ax = axes[i]
    ax.scatter(comp, metric, s=6, c="lightgray", label="archive")
    ax.scatter(comp[pf_mask], metric[pf_mask], s=20, c="red",
               edgecolors="k", linewidths=0.5, label="Pareto")
    ax.axvline(128, color="green", lw=0.5, ls="--")
    ax.set_xlim(60, 132)
    ax.set_xlabel("kvdim")
    ax.set_title(f"ans={s['ans']}, stride={s['stride']}\n"
                 f"iter={s['iters'][-1]}", fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("loss")
axes[0].legend(fontsize=7)
fig.tight_layout()
out = os.path.join(FIG_DIR, "final_pareto_scatter.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"[write] {out}")


# ---------- print short report ----------
print()
print("=" * 90)
print("KVDIM exploration summary (final iteration)")
print("=" * 90)
hdr = (f"{'label':<20s} {'ans':>4s} {'stride':>6s} {'#iter':>5s} "
       f"{'mean_kv':>7s} {'p95_kv':>6s} {'%kv128':>7s} "
       f"{'pf_max':>6s} {'pf_min':>6s} {'pf_n':>4s} {'pf_dL':>8s}")
print(hdr)
print("-" * len(hdr))
for r in summary_rows:
    print(f"{r['label']:<20s} {r['answer_tokens']:>4d} {r['stride']:>6d} "
          f"{r['n_iters']:>5d} {r['final_mean_kvdim']:>7.2f} "
          f"{r['final_p95_kvdim']:>6.1f} "
          f"{r['final_frac_layer_kvdim128']:>7.3f} "
          f"{r['final_pf_max_kvdim']:>6.1f} "
          f"{r['final_pf_min_kvdim']:>6.1f} "
          f"{r['final_pf_size']:>4d} "
          f"{r['final_pf_loss_spread']:>8.4f}")
