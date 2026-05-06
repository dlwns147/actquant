"""Read results.csv from run_corr.py, compute Spearman/Pearson correlations
and produce X_all-vs-Y / X_key-vs-Y scatter plots.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--save", type=str, required=True)
    args = ap.parse_args()

    rows = []
    with open(args.results) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise SystemExit("Empty results.csv")

    keys = rows[0].keys()
    Y_keys = [k for k in keys if k.startswith("Y_jsd_")]
    print("Y columns:", Y_keys)

    os.makedirs(args.save, exist_ok=True)

    summary = {}
    for ykey in Y_keys + ["Y_jsd_mean"]:
        if ykey not in keys:
            continue
        x_all = np.array([float(r["X_all"]) for r in rows])
        x_key = np.array([float(r["X_key"]) for r in rows])
        y = np.array([float(r[ykey]) for r in rows])
        mask = ~(np.isnan(x_all) | np.isnan(x_key) | np.isnan(y))
        if mask.sum() < 3:
            continue
        s_all, _ = spearmanr(x_all[mask], y[mask])
        s_key, _ = spearmanr(x_key[mask], y[mask])
        p_all, _ = pearsonr(x_all[mask], y[mask])
        p_key, _ = pearsonr(x_key[mask], y[mask])
        summary[ykey] = {
            "n": int(mask.sum()),
            "spearman_X_all": float(s_all),
            "spearman_X_key": float(s_key),
            "pearson_X_all": float(p_all),
            "pearson_X_key": float(p_key),
        }

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(x_all[mask], y[mask], s=20)
        axes[0].set_xlabel("X_all (JSD on all calib tokens)")
        axes[0].set_ylabel(ykey)
        axes[0].set_title(f"X_all vs {ykey}\nSpearman={s_all:.3f}  Pearson={p_all:.3f}")
        axes[1].scatter(x_key[mask], y[mask], s=20, color="tab:orange")
        axes[1].set_xlabel("X_key (JSD on key tokens)")
        axes[1].set_ylabel(ykey)
        axes[1].set_title(f"X_key vs {ykey}\nSpearman={s_key:.3f}  Pearson={p_key:.3f}")
        fig.tight_layout()
        out_png = os.path.join(args.save, f"scatter_{ykey}.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"  {ykey}: spearman X_all={s_all:.3f}  X_key={s_key:.3f}  → {out_png}")

    with open(os.path.join(args.save, "correlation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table for quick reading.
    md = ["| Y target | n | Spearman X_all | Spearman X_key | Pearson X_all | Pearson X_key |",
          "|---|---|---|---|---|---|"]
    for ykey, s in summary.items():
        md.append(f"| {ykey} | {s['n']} | {s['spearman_X_all']:.3f} | {s['spearman_X_key']:.3f} | {s['pearson_X_all']:.3f} | {s['pearson_X_key']:.3f} |")
    with open(os.path.join(args.save, "correlation_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("Wrote correlation_summary.json + correlation_table.md")


if __name__ == "__main__":
    main()
