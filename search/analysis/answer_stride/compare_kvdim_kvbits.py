"""Side-by-side: kvdim and kvbits right-end coverage (final iter) per ans."""
import csv
import os
import matplotlib.pyplot as plt

DAT = "/NAS/SJ/actquant/search/analysis/answer_stride/data"
FIG = "/NAS/SJ/actquant/search/analysis/answer_stride/figures_kvbits"

with open(f"{DAT}/summary.csv") as f:
    kvdim = list(csv.DictReader(f))
with open(f"{DAT}/summary_kvbits.csv") as f:
    kvb = list(csv.DictReader(f))

# Filter to ans=128, 512, 1024 with stride aligned (s=128 mostly).
def pick_kvdim(ans, stride):
    """final pf_max_kvdim normalized by 128 (right end)."""
    for r in kvdim:
        if int(r["answer_tokens"]) == ans and int(r["stride"]) == stride:
            return float(r["final_pf_max_kvdim"]) / 128, float(r["final_pf_loss_spread"])
    return None, None

def pick_kvb(ans, stride):
    for r in kvb:
        if int(r["ans"]) == ans and int(r["stride"]) == stride:
            return float(r["pf_max_kvbits"]) / 4.2, float(r["pf_loss_spread"])
    return None, None

ans_list = [128, 256, 512, 1024]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: PF max coverage (normalized to right end) at stride=128
ax = axes[0]
xs_kvdim = [128, 256, 512, 1024]
ys_kvdim = []
ys_kvb = []
for a in xs_kvdim:
    s = 32 if a == 128 else (64 if a == 256 else 128)  # use what's available for kvdim
    if a == 1024: s = 512
    yk, _ = pick_kvdim(a, s)
    ys_kvdim.append(yk)
    if s == 32 or s == 64:
        # for kvbits we only have s=128 (or s=512 at a=1024)
        yb, _ = pick_kvb(a, 128) if a < 1024 else pick_kvb(a, 512)
    else:
        yb, _ = pick_kvb(a, s)
    ys_kvb.append(yb)

ax.plot(xs_kvdim, ys_kvdim, "o-", label="kvdim", color="C0", lw=2, ms=10)
ax.plot(xs_kvdim, ys_kvb, "s-", label="kvbits", color="C2", lw=2, ms=10)
ax.set_xscale("log", base=2)
ax.set_xticks([128, 256, 512, 1024])
ax.set_xticklabels([128, 256, 512, 1024])
ax.set_xlabel("answer tokens")
ax.set_ylabel("PF max coverage (fraction of right end)")
ax.axhline(1.0, color="green", lw=0.5, ls="--", label="full right-end (=1)")
ax.set_ylim(0.5, 1.05)
ax.set_title("Right-end coverage:\nkvbits never collapses, kvdim collapses at small ans")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

# right: PF loss spread (raw) — illustrates that kvbits has SMALLER spread
# than kvdim but still gets full coverage (geometry, not SNR)
ax = axes[1]
strds = ["128 (legacy s=32 for ans=128)", "128", "128", "512"]
for label_, picker, color in [
    ("kvdim (PF spread)", pick_kvdim, "C0"),
    ("kvbits (PF spread)", pick_kvb, "C2"),
]:
    sp = []
    for a in xs_kvdim:
        s = 32 if a == 128 else (64 if a == 256 else 128)
        if a == 1024: s = 512
        if "kvbits" in label_:
            _, dl = pick_kvb(a, 128) if a < 1024 else pick_kvb(a, 512)
        else:
            _, dl = pick_kvdim(a, s)
        sp.append(dl)
    ax.plot(xs_kvdim, sp, "o-", label=label_, color=color, lw=2, ms=10)

ax.set_xscale("log", base=2)
ax.set_xticks([128, 256, 512, 1024])
ax.set_xticklabels([128, 256, 512, 1024])
ax.set_xlabel("answer tokens")
ax.set_ylabel("PF loss spread (max-min on Pareto)")
ax.set_title("PF loss spread is ABSOLUTELY SMALLER for kvbits;\n"
             "yet kvbits doesn't collapse — geometry beats spread")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

fig.tight_layout()
out = os.path.join(FIG, "kvdim_vs_kvbits_coverage.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"[write] {out}")
