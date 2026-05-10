"""Render all final figures from data/.

Outputs (figures/):
  ce_quant_cost.png        ΔL_proxy vs ΔL_gen (CE-based, fp16-relative)
  jsd_quant_cost.png       L_proxy_jsd vs L_gen_jsd (JSD against fp16)
  proxy_correlation.png    Pearson/Spearman bars · CE and JSD
  speed_vs_correlation.png Pareto trade-off (wall-clock vs JSD-Pearson)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ce = json.load(open(DATA / "ce_results.json"))
jsd = json.load(open(DATA / "jsd_results.json"))
fp16 = json.load(open(DATA / "fp16_baseline.json"))
speed = json.load(open(DATA / "speed_results.json")) if (DATA / "speed_results.json").exists() else None
ps_ce = json.load(open(DATA / "prefill_stride_ce.json")) if (DATA / "prefill_stride_ce.json").exists() else None
ps_jsd = json.load(open(DATA / "prefill_stride_jsd.json")) if (DATA / "prefill_stride_jsd.json").exists() else None

cfgs = sorted({r["cfg"] for r in ce}, key=lambda c: tuple(int(x) for x in c.split(":")))
seqlens = sorted({r["prompt_len"] for r in ce})
strides = sorted({int(s) for r in ce for s in r["strides"].keys()})
fp16_mean = {r["prompt_len"]: float(np.mean(r["fp16"])) for r in fp16}

PROXY_DISP = ["sgl0", "sglR"] + [f"s{s}" for s in strides]
PROXY_KEYS = [("single", None), ("single_r", None)] + [("strides", str(s)) for s in strides]
PROXY_COLORS = ["#888888", "#cc3333"] + list(plt.cm.viridis(np.linspace(0.05, 0.85, len(strides))))
PROXY_MARKER = {"sgl0": "X", "sglR": "*", **{f"s{s}": m for s, m in zip(strides, "osDh^")}}
CFG_COLOR = {c: plt.cm.viridis(i / max(1, len(cfgs) - 1)) for i, c in enumerate(cfgs)}


def get(rows, cfg, seq):
    for r in rows:
        if r["cfg"] == cfg and r["prompt_len"] == seq:
            return r
    return None


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


try:
    from scipy.stats import spearmanr
    sp = lambda a, b: float(spearmanr(a, b).correlation)
except Exception:
    sp = lambda a, b: float("nan")


def proxy_means(rows, seq):
    """Return {proxy_name: np.array of len(cfgs)} of mean values."""
    out = {"gen": np.array([np.mean(get(rows, c, seq)["gen"]) for c in cfgs]),
           "sgl0": np.array([np.mean(get(rows, c, seq)["single"]) for c in cfgs]),
           "sglR": np.array([np.mean(get(rows, c, seq)["single_r"]) for c in cfgs])}
    for st in strides:
        out[f"s{st}"] = np.array([np.mean(get(rows, c, seq)["strides"][str(st)]) for c in cfgs])
    return out


# ── Fig 1: CE quant cost (ΔL = L − L_fp16) ────────────────────────────
fig, axes = plt.subplots(1, len(seqlens), figsize=(8 * len(seqlens), 7))
for ax, seq in zip(axes, seqlens):
    m = proxy_means(ce, seq)
    base = fp16_mean[seq]
    Δ = {k: v - base for k, v in m.items()}
    lo = min(d.min() for d in Δ.values()) - 0.05
    hi = max(d.max() for d in Δ.values()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    for i, c in enumerate(cfgs):
        col = CFG_COLOR[c]
        for label in PROXY_DISP:
            ax.scatter(Δ["gen"][i], Δ[label][i], marker=PROXY_MARKER[label],
                       s=70 if label != "sglR" else 120, color=col,
                       edgecolor="black", lw=0.4)
        ax.annotate(f"K{c.replace(':','V')}", (Δ["gen"][i], Δ["sgl0"][i]),
                    fontsize=8, xytext=(5, -3), textcoords="offset points", color=col)
    ax.set_xlabel(r"$\Delta L_{gen}$ = L_gen − L_fp16  (real CE quant cost, nat)")
    ax.set_ylabel(r"$\Delta L_{proxy}$  (nat)")
    ax.set_title(f"seq = {seq}  (fp16 = {base:.3f})")
    ax.grid(alpha=0.3)

elements = [Line2D([], [], marker=PROXY_MARKER[p], color="w", markerfacecolor="gray",
                   markersize=11 if p == "sglR" else 9, label=p) for p in PROXY_DISP]
axes[-1].legend(handles=elements, loc="upper left", fontsize=9)
fig.suptitle("CE: ΔL_proxy vs ΔL_gen across 9 ≤4-bit configs (diagonal = perfect proxy)")
plt.tight_layout()
plt.savefig(FIG / "ce_quant_cost.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIG / 'ce_quant_cost.png'}")

# ── Fig 2: JSD quant cost (L_proxy_jsd vs L_gen_jsd directly) ─────────
fig, axes = plt.subplots(1, len(seqlens), figsize=(8 * len(seqlens), 7))
for ax, seq in zip(axes, seqlens):
    m = proxy_means(jsd, seq)
    lo = min(v.min() for v in m.values()) * 0.9
    hi = max(v.max() for v in m.values()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    for i, c in enumerate(cfgs):
        col = CFG_COLOR[c]
        for label in PROXY_DISP:
            ax.scatter(m["gen"][i], m[label][i], marker=PROXY_MARKER[label],
                       s=70 if label != "sglR" else 120, color=col,
                       edgecolor="black", lw=0.4)
        ax.annotate(f"K{c.replace(':','V')}", (m["gen"][i], m["sgl0"][i]),
                    fontsize=8, xytext=(5, -3), textcoords="offset points", color=col)
    ax.set_xlabel("L_gen JSD  (real-decode JSD vs fp16)")
    ax.set_ylabel("L_proxy JSD")
    ax.set_title(f"seq = {seq}")
    ax.grid(alpha=0.3)

axes[-1].legend(handles=elements, loc="upper left", fontsize=9)
fig.suptitle("JSD: L_proxy vs L_gen across 9 ≤4-bit configs (diagonal = perfect proxy)")
plt.tight_layout()
plt.savefig(FIG / "jsd_quant_cost.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIG / 'jsd_quant_cost.png'}")

# ── Fig 3: Pearson/Spearman bars (CE row + JSD row) ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for col_idx, (rows, metric_name) in enumerate(zip([ce, jsd], ["CE", "JSD"])):
    for row_idx, (corr_name, fn) in enumerate(zip(["Pearson", "Spearman"], [corr, sp])):
        ax = axes[row_idx, col_idx]
        x = np.arange(len(seqlens))
        w = 0.13
        for i, ((label, (k, s)), col) in enumerate(zip(zip(PROXY_DISP, PROXY_KEYS), PROXY_COLORS)):
            ys = []
            for seq in seqlens:
                g = np.array([np.mean(get(rows, c, seq)["gen"]) for c in cfgs])
                if s is None:
                    p = np.array([np.mean(get(rows, c, seq)[k]) for c in cfgs])
                else:
                    p = np.array([np.mean(get(rows, c, seq)[k][s]) for c in cfgs])
                ys.append(fn(g, p))
            ax.bar(x + (i - len(PROXY_KEYS)/2 + 0.5) * w, ys, w,
                   label=label, color=col, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seqlens])
        ax.set_xlabel("seqlen")
        ax.set_ylabel(f"{corr_name}(L_gen, L_proxy)")
        ax.set_title(f"{metric_name}: {corr_name} over 9 configs")
        ax.set_ylim(0.5, 1.01)
        ax.grid(alpha=0.3, axis="y")
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8, ncol=3)

plt.tight_layout()
plt.savefig(FIG / "proxy_correlation.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIG / 'proxy_correlation.png'}")

# ── Fig 4: Speed vs JSD Pearson (only if speed data exists) ───────────
if speed is not None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    speed_by_seq = {r["prompt_len"]: r["ms"] for r in speed}
    timing_key = {"sgl0": "single_0", "sglR": "single_R",
                  **{f"s{s}": f"stride_{s}" for s in strides}}
    seq_to_size = {2048: 110, 4096: 170, 8192: 240}
    proxy_styles = {"sgl0": ("X", "#888888"), "sglR": ("*", "#cc3333"),
                    **{f"s{s}": ("o", c) for s, c in zip(strides, plt.cm.viridis(np.linspace(0.05, 0.85, len(strides))))}}

    # Connect proxies of the same seqlen
    for seq in seqlens:
        if seq not in speed_by_seq:
            continue
        xs, ys = [], []
        for label in PROXY_DISP:
            x_ms = speed_by_seq[seq][timing_key[label]]
            g = np.array([np.mean(get(jsd, c, seq)["gen"]) for c in cfgs])
            k, s = (("single", None) if label == "sgl0"
                    else ("single_r", None) if label == "sglR"
                    else ("strides", label[1:]))
            p = np.array([np.mean(get(jsd, c, seq)[k]) if s is None
                          else np.mean(get(jsd, c, seq)[k][s]) for c in cfgs])
            xs.append(x_ms); ys.append(corr(g, p))
        order = np.argsort(xs)
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                "-", lw=0.7, alpha=0.4, color="black")

    for label in PROXY_DISP:
        marker, color = proxy_styles[label]
        for seq in seqlens:
            if seq not in speed_by_seq:
                continue
            x_ms = speed_by_seq[seq][timing_key[label]]
            g = np.array([np.mean(get(jsd, c, seq)["gen"]) for c in cfgs])
            k, s = (("single", None) if label == "sgl0"
                    else ("single_r", None) if label == "sglR"
                    else ("strides", label[1:]))
            p = np.array([np.mean(get(jsd, c, seq)[k]) if s is None
                          else np.mean(get(jsd, c, seq)[k][s]) for c in cfgs])
            r = corr(g, p)
            ax.scatter(x_ms, r, s=seq_to_size.get(seq, 90), marker=marker, color=color,
                       edgecolor="black", lw=0.6, zorder=3)
            ax.annotate(f"{label}·{seq//1024}k", (x_ms, r),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)

    # L_gen baseline points
    for seq in seqlens:
        if seq in speed_by_seq:
            ax.scatter(speed_by_seq[seq]["gen"], 1.0,
                       s=seq_to_size.get(seq, 90)*0.8, marker="P", color="gold",
                       edgecolor="black", lw=1, zorder=4)
            ax.annotate(f"gen·{seq//1024}k", (speed_by_seq[seq]["gen"], 1.0),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("wall-clock (ms, log scale) · 1 sample, K2V2")
    ax.set_ylabel("Pearson(L_gen JSD, L_proxy JSD) over 9 configs")
    ax.set_title("Speed vs JSD-fidelity Pareto\n"
                 "size: small=2k, mid=4k, large=8k · gold ⊕ = real-decode reference")
    ax.grid(alpha=0.3, which="both")
    elements2 = [Line2D([], [], marker=PROXY_MARKER[p], color="w",
                        markerfacecolor=PROXY_COLORS[i if i < 2 else i],
                        markersize=11 if p == "sglR" else 9, label=p)
                 for i, p in enumerate(PROXY_DISP)]
    ax.legend(handles=elements2, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "speed_vs_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {FIG / 'speed_vs_correlation.png'}")
else:
    print("(skip speed plot — data/speed_results.json missing)")

# ── Fig 5: prefill+answer-stride bias vs stride ───────────────────────
# Restricted to answer_len=128 to keep this panel comparable across configs;
# longer answer_len analysis lives in 07_plot_long_answer.py.
if ps_ce is not None and ps_jsd is not None:
    PS_ANS_LEN = 128
    ps_strides = sorted({int(s) for r in ps_ce
                         if r.get("answer_len", 128) == PS_ANS_LEN
                         for s in r["strides"].keys()})

    def get_ps(rows, cfg, seq):
        for r in rows:
            if (r["cfg"] == cfg and r["prompt_len"] == seq
                    and r.get("answer_len", 128) == PS_ANS_LEN):
                return r
        return None

    fig, axes = plt.subplots(2, len(seqlens), figsize=(8 * len(seqlens), 12))
    for col_idx, seq in enumerate(seqlens):
        # Reference: L_gen for this (cfg, seq)
        gen_ce = {c: np.mean(get(ce, c, seq)["gen"]) for c in cfgs}
        gen_jsd = {c: np.mean(get(jsd, c, seq)["gen"]) for c in cfgs}
        full_str_jsd128 = {c: np.mean(get(jsd, c, seq)["strides"]["128"]) for c in cfgs}

        # CE delta vs gen
        ax = axes[0, col_idx]
        for c in cfgs:
            r = get_ps(ps_ce, c, seq)
            if r is None: continue
            ys = [np.mean(r["strides"][str(s)]) - gen_ce[c] for s in ps_strides]
            ax.plot(ps_strides, ys, "-o", color=CFG_COLOR[c], lw=1.6,
                    ms=7, label=f"K{c.replace(':','V')}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks(ps_strides)
        ax.set_xticklabels([str(s) for s in ps_strides])
        ax.set_xlabel("answer-stride")
        ax.set_ylabel(r"$\Delta$CE = mean(L_prefill_stride) − mean(L_gen)")
        ax.set_title(f"seq = {seq} · CE bias vs answer-stride\n(prefill prompt → stride answer)")
        ax.grid(alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper left")

        # JSD vs gen, also overlay full-seq stride@128 reference
        ax = axes[1, col_idx]
        for c in cfgs:
            r = get_ps(ps_jsd, c, seq)
            if r is None: continue
            ys = [np.mean(r["strides"][str(s)]) - gen_jsd[c] for s in ps_strides]
            ax.plot(ps_strides, ys, "-o", color=CFG_COLOR[c], lw=1.6, ms=7,
                    label=f"K{c.replace(':','V')}")
            # full-seq stride@128 baseline (horizontal dashed)
            ax.axhline(full_str_jsd128[c] - gen_jsd[c],
                       color=CFG_COLOR[c], linestyle=":", lw=0.7, alpha=0.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks(ps_strides)
        ax.set_xticklabels([str(s) for s in ps_strides])
        ax.set_xlabel("answer-stride")
        ax.set_ylabel(r"$\Delta$JSD = mean(L_prefill_stride) − mean(L_gen)")
        ax.set_title(f"seq = {seq} · JSD bias vs answer-stride\n"
                     "(dotted: full-seq stride@128 baseline for comparison)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG / "prefill_stride_bias.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {FIG / 'prefill_stride_bias.png'}")

    # Correlation table — only over (cfg, seq) cells where prefill_stride data exists
    ps_cfgs = [c for c in cfgs if get_ps(ps_jsd, c, seqlens[0]) is not None]
    if ps_cfgs:
        print(f"\n=== Prefill-stride: config-level Pearson(L_gen, L_prefill_stride) "
              f"over {len(ps_cfgs)} cfgs at A={PS_ANS_LEN} ===")
        print("--- JSD ---")
        print(f"{'seq':>5}  " + "  ".join([f"s{s:>4}" for s in ps_strides]))
        for seq in seqlens:
            present = [c for c in ps_cfgs if get_ps(ps_jsd, c, seq) is not None]
            if not present:
                continue
            line = f"{seq:>5}  "
            g = np.array([np.mean(get(jsd, c, seq)["gen"]) for c in present])
            for s in ps_strides:
                p = np.array([np.mean(get_ps(ps_jsd, c, seq)["strides"][str(s)])
                              for c in present])
                line += f"{corr(g, p):+7.4f}  "
            print(line)
else:
    print("(skip prefill-stride plot — data/prefill_stride_*.json missing)")

# ── Print summary tables ──────────────────────────────────────────────
print("\n=== JSD config-level Pearson(L_gen, L_proxy) over 9 configs ===")
print(f"{'seq':>5}  " + "  ".join([f"{p:>7}" for p in PROXY_DISP]))
for seq in seqlens:
    line = f"{seq:>5}  "
    g = np.array([np.mean(get(jsd, c, seq)["gen"]) for c in cfgs])
    for label, (k, s) in zip(PROXY_DISP, PROXY_KEYS):
        if s is None:
            p = np.array([np.mean(get(jsd, c, seq)[k]) for c in cfgs])
        else:
            p = np.array([np.mean(get(jsd, c, seq)[k][s]) for c in cfgs])
        line += f"{corr(g, p):+7.4f}  "
    print(line)
