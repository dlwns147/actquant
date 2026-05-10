"""
Speed (iter_t) vs right-end coverage (PF max kvdim) scatter, colored by
Pareto loss spread. One marker per existing run.
"""
import os
import csv
import matplotlib.pyplot as plt

OUT_DIR = "/NAS/SJ/actquant/search/analysis/answer_stride/figures"

with open("/NAS/SJ/actquant/search/analysis/answer_stride/data/summary.csv") as f:
    summary = list(csv.DictReader(f))
with open("/NAS/SJ/actquant/search/analysis/answer_stride/data/timings.csv") as f:
    timings = {r["label"].rstrip(): r for r in csv.DictReader(f)}

# Map summary labels (analysis_v3 names) to timings labels.
LABEL_MAP = {
    "pp128_s32": "pp128_s32",
    "pp256_s64": "pp256_s64",
    "pp256_s128": "pp256_s128",
    "pp512_s128_s2048": "pp512_s128_seq2k",
    "pp512_s128": "pp512_s128_n32",
    "pp512_s128_n64": "pp512_s128_n64",
    "pp1024_s512": "pp1024_s512",
    "pp512_s64_v2": "pp512_s64_v2",
    "pp256_s128_resume_v2": "pp256_s128_resume_v2",
    "pp1024_s128_v2_fixed": "pp1024_s128_v2_fixed",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for s in summary:
    t = timings.get(LABEL_MAP[s["label"]])
    if not t:
        continue
    x = float(t["avg_iter_time_s"])
    y = float(s["final_pf_max_kvdim"])
    spread = float(s["final_pf_loss_spread"])
    sc = ax.scatter(x, y, s=300, c=spread, cmap="viridis",
                    vmin=0, vmax=0.5, edgecolors="k", zorder=3)
    ax.annotate(f"a={s['answer_tokens']},s={s['stride']}\n"
                f"n={t['n_sample']}",
                xy=(x, y), xytext=(6, 4),
                textcoords="offset points", fontsize=8)
ax.axhline(128, color="green", lw=0.5, ls="--", label="kvdim=128 (no prune)")
ax.axhline(64, color="red", lw=0.5, ls="--", label="kvdim=64 (50% prune)")
ax.set_xlabel("avg iter time (s)  ←  faster")
ax.set_ylabel("Pareto front max kvdim  ↑  better right-end coverage")
ax.set_xlim(0, 70)
ax.set_ylim(60, 132)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Speed vs right-end coverage  (color = PF loss spread)")
plt.colorbar(sc, ax=ax, label="PF loss spread (max - min)")

# right plot: PF loss spread vs pf_max_kvdim (driver of right-end loss)
ax = axes[1]
for s in summary:
    spread = float(s["final_pf_loss_spread"])
    y = float(s["final_pf_max_kvdim"])
    ax.scatter(spread, y, s=160, c="C0", edgecolors="k", zorder=3)
    ax.annotate(f"a={s['answer_tokens']},s={s['stride']}",
                xy=(spread, y), xytext=(6, 4),
                textcoords="offset points", fontsize=8)
ax.axhline(128, color="green", lw=0.5, ls="--")
ax.axhline(64, color="red", lw=0.5, ls="--")
ax.set_xlabel("PF loss spread (max-min)  →  higher = more discriminable")
ax.set_ylabel("PF max kvdim")
ax.set_xlim(0, 0.5)
ax.set_ylim(60, 132)
ax.grid(alpha=0.3)
ax.set_title("Loss spread is the driver of right-end retention")

fig.tight_layout()
out = os.path.join(OUT_DIR, "speed_vs_coverage.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"[write] {out}")
