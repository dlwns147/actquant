"""
Diagnose noise floor of JSD across architectures (within complexity bins) and
quantify forward-count / chunk-overhead cost as a function of (ans, stride).
"""
import os
import json
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

THINK_DIR = "/NAS/SJ/actquant/search/save/search/think"
FIG = "/NAS/SJ/actquant/search/analysis/answer_stride/figures"

# Pick representative final-iter snapshots, use same iter from each.
# (label, dir_glob, ans, stride, ntoken, prefill_estimate, n_sample, last_iter)
RUNS = [
    ("pp128_s32",   "2605071157_*128stride_pp128",  128, 32,  32,  1792, "iter_150.stats"),
    ("pp256_s64",   "2605070434_*64stride_pp256",   256, 64,  64,  2048, "iter_120.stats"),
    ("pp256_s128",  "2605071929_*128stride_pp256",  256, 128, 32,  1536, "iter_150.stats"),  # extended
    ("pp512_s64",   "2605071929_*64stride_pp512",   512, 64,  32,  1024, "iter_150.stats"),
    ("pp512_s128",  "2605071156_*128stride_pp512",  512, 128, 32,  1024, "iter_90.stats"),
    ("pp512_s128n64","2605070643_*128stride_pp512", 512, 128, 64,  2048, "iter_150.stats"),
    ("pp1024_s512", "2605070803_*512stride_pp1024",1024, 512, 64,  1024, "iter_130.stats"),
]


def load(run_glob, fname):
    matches = glob.glob(os.path.join(THINK_DIR, run_glob, fname))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def kvdim_per_arch(arch):
    pk = np.array(arch["p"]["k"], dtype=np.int32)
    pv = np.array(arch["p"]["v"], dtype=np.int32)
    rem = 128 - np.maximum(pk, pv)
    return rem.mean()  # per-arch average remained dim


# ---------------- 1) Noise floor (within-comp std) -----------------
print("=" * 95)
print("Within-complexity-bin metric STD: rough estimate of evaluation noise")
print(f"{'label':<13} {'ans':>4} {'stride':>6} {'mean_metric':>12} "
      f"{'global_std':>10} {'within_bin_std':>14} {'noise/spread':>12}")
print("-" * 95)
noise_table = []
for label, gl, ans, stride, nsamp, prefill, last in RUNS:
    d = load(gl, last)
    if d is None:
        print(f"{label}: skip (not found)")
        continue
    metric = np.array([x[1] for x in d["archive"]])
    comp = np.array([x[2] for x in d["archive"]])

    # bin comp into 8 bins, take std within each bin (architectures with similar
    # average-kvdim but different per-block patterns), avg the within-bin std
    bins = np.linspace(comp.min(), comp.max() + 1e-6, 9)
    inds = np.digitize(comp, bins) - 1
    within_stds = []
    for b in range(8):
        m = metric[inds == b]
        if len(m) >= 5:
            within_stds.append(m.std())
    within_std = float(np.mean(within_stds)) if within_stds else float("nan")
    global_std = metric.std()
    spread = metric.max() - metric.min()
    nratio = within_std / spread if spread > 0 else float("nan")
    print(f"{label:<13} {ans:>4} {stride:>6} {metric.mean():>12.5f} "
          f"{global_std:>10.5f} {within_std:>14.5f} {nratio:>12.3f}")
    noise_table.append((label, ans, stride, nsamp, within_std, spread))

# ---------------- 2) Forward-count & cost model -----------------
print()
print("=" * 95)
print("Forward count and rough cost model (per architecture eval)")
print(f"{'label':<13} {'ans':>4} {'stride':>6} {'fwd/s':>5} {'prefill':>7} "
      f"{'attn_cost':>10} {'wall_iter(s)':>12}")
print("-" * 95)

ITER_TIME = {
    "pp128_s32": 37.7, "pp256_s64": 59.0, "pp256_s128": 23.1,
    "pp512_s64": 38.6, "pp512_s128": 19.8, "pp512_s128n64": 21.5,
    "pp1024_s512": 37.8,
}
for label, gl, ans, stride, nsamp, prefill, last in RUNS:
    n_chunks = math.ceil(ans / stride)
    fwd_per_sample = 1 + n_chunks
    # rough attention cost proxy:
    # prefill self-attn: prefill^2
    # answer chunks: each chunk i has length=stride attending to (prefill + i*stride)
    L_p = prefill
    L_a = ans
    s = stride
    cost = L_p ** 2 / 2  # prefill self-attn
    for i in range(n_chunks):
        chunk_len = min(s, L_a - i * s)
        kv_len = L_p + i * s
        cost += chunk_len * kv_len + chunk_len ** 2 / 2
    print(f"{label:<13} {ans:>4} {stride:>6} {fwd_per_sample:>5} "
          f"{prefill:>7d} {cost/1e6:>10.2f}M {ITER_TIME.get(label, 0):>12.1f}")

# ---------------- 3) noise/spread vs ans plot -----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
groups = {}
for lbl, ans, stride, nsamp, ns, sp in noise_table:
    groups.setdefault(stride, []).append((ans, ns, sp, nsamp, lbl))
for stride, pts in sorted(groups.items()):
    pts.sort()
    xs = [p[0] for p in pts]
    noise = [p[1] for p in pts]
    spread = [p[2] for p in pts]
    ax.plot(xs, noise, "o-", label=f"within-bin σ (s={stride})", alpha=0.85)
    ax.plot(xs, spread, "s--", label=f"PF spread (s={stride})", alpha=0.6)
ax.set_xlabel("answer tokens")
ax.set_ylabel("JSD")
ax.set_xscale("log", base=2)
ax.set_xticks([128, 256, 512, 1024])
ax.set_xticklabels([128, 256, 512, 1024])
ax.set_title("Noise (within-comp σ) vs PF spread — \n"
             "spread must exceed noise to discriminate kvdim=128")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left")

# subplot 2: signal-to-noise ratio
ax = axes[1]
sigma_x_n = []
spread_x_n = []
for lbl, ans, stride, nsamp, ns, sp in noise_table:
    snr = sp / ns if ns > 0 else float("nan")
    ax.scatter(ans, snr, s=80,
               label=f"a={ans} s={stride}",
               c=("C0" if stride == 128 else "C1" if stride == 64
                  else "C2" if stride == 32 else "C3"))
    ax.annotate(f"s={stride}", (ans, snr), xytext=(5, 3),
                textcoords="offset points", fontsize=8)
ax.axhline(2, color="red", lw=0.8, ls="--", label="SNR=2 (signal=2σ)")
ax.set_xlabel("answer tokens")
ax.set_ylabel("PF spread / within-bin σ  (signal-to-noise)")
ax.set_xscale("log", base=2)
ax.set_xticks([128, 256, 512, 1024])
ax.set_xticklabels([128, 256, 512, 1024])
ax.set_title("Loss SNR vs answer length — driver of right-end retention")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left")

fig.tight_layout()
out = os.path.join(FIG, "noise_vs_signal.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"\n[write] {out}")

# ---------------- 4) cost vs stride plot for fixed ans=1024 -----------
print()
print("Predicted iter time for ans=1024, n_sample=32, varying stride")
print("(extrapolated from 4.4ms/forward measured at 9-fwd/sample)")
ms_per_fwd = 4.4
n_archs = 30
nsamp = 32
overhead_per_iter = 5  # predictor+GA
print(f"{'stride':>8} {'fwd/s':>5} {'fwd/iter':>10} {'iter_t(s)':>10} "
      f"{'150iter(min)':>14}")
print("-" * 55)
for s in [64, 128, 256, 512, 1024]:
    nch = math.ceil(1024 / s)
    fwd_per_s = 1 + nch
    fwd_per_iter = fwd_per_s * nsamp * n_archs
    t = fwd_per_iter * ms_per_fwd / 1000 + overhead_per_iter
    total_min = t * 150 / 60
    print(f"{s:>8} {fwd_per_s:>5} {fwd_per_iter:>10} {t:>10.1f} {total_min:>14.1f}")
