"""Plot predicted iter time vs stride for various answer lengths."""
import os
import math
import matplotlib.pyplot as plt

FIG = "/NAS/SJ/actquant/search/analysis/answer_stride/figures"

ms_per_fwd = 4.4   # measured at ans>=512 prefill_prompt
n_archs = 30
nsamp = 32
overhead_per_iter = 5
total_iters = 150

ans_list = [512, 1024, 2048]
strides = [32, 64, 128, 256, 512, 1024, 2048]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ans in ans_list:
    xs, ts = [], []
    for s in strides:
        if s > ans:
            continue
        nch = math.ceil(ans / s)
        fwd_per_s = 1 + nch
        fwd_per_iter = fwd_per_s * nsamp * n_archs
        t = fwd_per_iter * ms_per_fwd / 1000 + overhead_per_iter
        xs.append(s); ts.append(t)
    axes[0].plot(xs, ts, "o-", label=f"ans={ans}", lw=2)
    axes[1].plot(xs, [t * total_iters / 60 for t in ts], "o-",
                 label=f"ans={ans}", lw=2)

for ax in axes:
    ax.set_xlabel("stride")
    ax.set_xscale("log", base=2)
    ax.set_xticks(strides)
    ax.set_xticklabels([str(s) for s in strides])
    ax.axvline(128, color="green", lw=0.7, ls="--",
               label="residual_length=128")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

axes[0].set_ylabel("avg iter time (s)")
axes[0].set_title("Predicted iter time vs stride")
axes[1].set_ylabel("Total search time (min)")
axes[1].set_title(f"Total {total_iters}-iter search time")
fig.tight_layout()
out = os.path.join(FIG, "cost_vs_stride.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"[write] {out}")

# Also: speed-up factor of stride=512 vs stride=128 for each ans
print()
print("Speedup of stride=512 vs stride=128 (kv_proxy_study: equal-ranking, "
      "Pearson > 0.99):")
for ans in [256, 512, 1024, 2048]:
    fwd_128 = 1 + math.ceil(ans / 128)
    fwd_512 = 1 + math.ceil(ans / 512)
    print(f"  ans={ans:>4}: fwd 128→{fwd_128}, 512→{fwd_512}, "
          f"speedup={fwd_128/fwd_512:.2f}x")
