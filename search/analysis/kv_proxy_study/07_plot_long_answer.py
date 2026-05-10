"""Plot prefill+stride bias as a function of stride, faceted by answer_len.

Reads data/prefill_stride_{ce,jsd}.json which now contains rows keyed by
(cfg, prompt_len, answer_len). For each panel (one per answer_len) we plot
Δ = mean(L_proxy_stride_S) − mean(L_proxy_stride_2) per config × stride
(using stride=2 as a near-perfect L_gen proxy, verified to be <0.001 nat off).

Hypothesis: at fixed stride/R ratio, bias should be ≈ independent of answer_len
because non-boundary chunks see KV state matching real decode and only chunks
crossing the residual boundary contribute bias proportional to stride.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ce = json.load(open(DATA / "prefill_stride_ce.json"))
jsd = json.load(open(DATA / "prefill_stride_jsd.json"))

answer_lens = sorted({r["answer_len"] for r in ce})
prompt_lens = sorted({r["prompt_len"] for r in ce})
cfgs = sorted({r["cfg"] for r in ce}, key=lambda c: tuple(int(x) for x in c.split(":")))
all_strides = sorted({int(s) for r in ce for s in r["strides"].keys()})

cfg_color = {c: plt.cm.viridis(i / max(1, len(cfgs) - 1)) for i, c in enumerate(cfgs)}


def get(rows, cfg, prompt, ans):
    for r in rows:
        if r["cfg"] == cfg and r["prompt_len"] == prompt and r["answer_len"] == ans:
            return r
    return None


# ── Print bias table ──────────────────────────────────────────────────
print("=== prefill+stride: ΔJSD vs stride=2 baseline (proxy for L_gen) ===")
P = prompt_lens[0]  # use first prompt_len (typically 2048)
header = f"{'cfg':>5} {'A':>5}  " + "  ".join([f"s{s:>4}" for s in all_strides])
print(header)
for cfg in cfgs:
    for A in answer_lens:
        r = get(jsd, cfg, P, A)
        if r is None: continue
        baseline_strides = [s for s in r["strides"] if int(s) <= A]
        s2 = "2" if "2" in r["strides"] else baseline_strides[0]
        ref = np.mean(r["strides"][s2])
        line = f"{cfg:>5} {A:>5}  "
        for st in all_strides:
            if str(st) in r["strides"] and st <= A:
                line += f"{np.mean(r['strides'][str(st)]) - ref:+7.5f}  "
            else:
                line += f"{'-':>7}  "
        print(line)

# ── Figure: ΔJSD vs stride, panel per answer_len, line per config ─────
fig, axes = plt.subplots(1, len(answer_lens), figsize=(5 * len(answer_lens), 5.5),
                         sharey=True)
if len(answer_lens) == 1:
    axes = [axes]

for ax, A in zip(axes, answer_lens):
    for cfg in cfgs:
        r = get(jsd, cfg, P, A)
        if r is None: continue
        s2 = "2" if "2" in r["strides"] else None
        if s2 is None: continue
        ref = np.mean(r["strides"][s2])
        xs, ys = [], []
        for st in all_strides:
            if str(st) in r["strides"] and st <= A:
                xs.append(st)
                ys.append(np.mean(r["strides"][str(st)]) - ref)
        if xs:
            ax.plot(xs, ys, "-o", color=cfg_color[cfg], lw=1.5, ms=6,
                    label=f"K{cfg.replace(':','V')}")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(128, color="red", lw=0.7, alpha=0.4, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("answer-stride")
    if A == answer_lens[0]:
        ax.set_ylabel(r"$\Delta$JSD vs stride=2 baseline")
    ax.set_title(f"answer_len = {A}\n(R=128 marked)")
    ax.grid(alpha=0.3)
    if A == answer_lens[0]:
        ax.legend(fontsize=8, loc="upper left")

fig.suptitle(f"prefill+stride: bias vs stride at varying answer_len  (prompt={P})",
             y=1.02)
plt.tight_layout()
out = FIG / "prefill_stride_long_answer.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")

# ── Bias vs stride/R ratio (collapse onto single curve) ───────────────
fig, ax = plt.subplots(figsize=(8, 6))
R = 128
A_marker = {a: m for a, m in zip(answer_lens, "osDh^X")}
for cfg in cfgs:
    for A in answer_lens:
        r = get(jsd, cfg, P, A)
        if r is None: continue
        s2 = "2" if "2" in r["strides"] else None
        if s2 is None: continue
        ref = np.mean(r["strides"][s2])
        xs, ys = [], []
        for st in all_strides:
            if str(st) in r["strides"] and st <= A:
                xs.append(st / R)
                ys.append(np.mean(r["strides"][str(st)]) - ref)
        if xs:
            ax.plot(xs, ys, "-", color=cfg_color[cfg], lw=1, alpha=0.7,
                    marker=A_marker[A], ms=6)
ax.axhline(0, color="black", lw=0.7)
ax.set_xscale("log", base=2)
ax.set_xlabel("stride / R   (R=128)")
ax.set_ylabel(r"$\Delta$JSD vs stride=2 baseline")
ax.set_title("Bias collapses on stride/R ratio across answer_len\n"
             "(color = config, marker = answer_len)")
ax.grid(alpha=0.3)

from matplotlib.lines import Line2D
elements = [Line2D([], [], marker="o", color="w", markerfacecolor=cfg_color[c],
                   markersize=9, label=f"K{c.replace(':','V')}") for c in cfgs]
elements += [Line2D([], [], marker=A_marker[a], color="gray", markerfacecolor="w",
                    markersize=9, label=f"A={a}") for a in answer_lens]
ax.legend(handles=elements, fontsize=8, ncol=2)
plt.tight_layout()
out2 = FIG / "prefill_stride_bias_collapse.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out2}")
