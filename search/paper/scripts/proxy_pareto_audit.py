"""Reproduce the proxy-boundary statistics reported in paper/main.md.

The script is GPU-free and read-only.  It uses paired post-hoc HQQ labels from
the current production AWQ archive and the latest narrow-budget allocation
cells.  Pareto objectives are (loss, average W bits, effective KV bits), all
minimized.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr


ROOT = "/NAS/SJ/actquant/search"
RUN = os.path.join(
    ROOT,
    "save/second_search",
    "2607281255_Llama-3.1-8B-Instruct_joint_awq_kivi_think_"
    "ard_gpplstyp_doe100_it15n20p200_subset-st-ckv102d_eps0.05_"
    "dk0_st128_pp512_sk8_s0",
)
ALLOC = os.path.join(ROOT, "tests", "awq_alloc_flip")


def json_lines(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def load_production():
    with open(os.path.join(RUN, "iter_15.stats")) as handle:
        archive = json.load(handle)["archive"]
    proxy = {}
    for row in json_lines(os.path.join(RUN, "y_hqq_s*.jsonl")):
        proxy[row["idx"]] = row["y_hqq"]
    indices = sorted(proxy)
    target_loss = np.array([archive[i][1] for i in indices])
    proxy_loss = np.array([proxy[i] for i in indices])
    wbits = np.array([archive[i][2] for i in indices])
    kvbits = np.array([archive[i][3] for i in indices])
    return target_loss, proxy_loss, wbits, kvbits


def pareto_front(loss, wbits, kvbits, tolerance=0.0):
    """Return indices nondominated under the repository's JSD tie rule."""
    front = set()
    for i in range(len(loss)):
        dominates = (
            (loss <= loss[i] + tolerance)
            & (wbits <= wbits[i])
            & (kvbits <= kvbits[i])
            & (
                (loss < loss[i] - tolerance)
                | (wbits < wbits[i])
                | (kvbits < kvbits[i])
            )
        )
        dominates[i] = False
        if not dominates.any():
            front.add(i)
    return front


def jaccard(left, right):
    return len(left & right) / max(len(left | right), 1)


def load_narrow_cells():
    cells = []
    for prefix in ("discv2", "discv2grid", "discv3"):
        with open(os.path.join(ALLOC, f"{prefix}_grid_specs.json")) as handle:
            spec = json.load(handle)
        level = {block["bid"]: block["level"] for block in spec["w_blocks"]}
        valid_kv = {
            block["bid"]
            for block in spec["kv_blocks"]
            if block.get("role", "ctx") != "kvpair"
        }
        target = {
            (row["wb"], row["kb"]): row["y"]
            for row in json_lines(os.path.join(ALLOC, f"{prefix}_awq_s*.jsonl"))
        }
        proxy = {
            (row["wb"], row["kb"]): row["y"]
            for row in json_lines(os.path.join(ALLOC, f"{prefix}_hqq_s*.jsonl"))
        }
        grouped = {}
        for key, target_value in target.items():
            wblock, kvblock = key
            if kvblock in valid_kv and key in proxy and level[wblock] >= 2.4:
                grouped.setdefault((level[wblock], kvblock), []).append(
                    (proxy[key], target_value)
                )
        cells.extend(values for values in grouped.values() if len(values) >= 6)
    return cells


def main():
    target, proxy, wbits, kvbits = load_production()
    exact_target = pareto_front(target, wbits, kvbits)
    exact_proxy = pareto_front(proxy, wbits, kvbits)
    exact_intersection = exact_target & exact_proxy

    certificate_count = 0
    for i in exact_target - exact_proxy:
        witness = (
            (proxy <= proxy[i])
            & (wbits <= wbits[i])
            & (kvbits <= kvbits[i])
            & ((proxy < proxy[i]) | (wbits < wbits[i]) | (kvbits < kvbits[i]))
            & (target > target[i])
        )
        witness[i] = False
        certificate_count += int(witness.any())

    tolerance = 1e-3
    tolerant_target = pareto_front(target, wbits, kvbits, tolerance)
    tolerant_proxy = pareto_front(proxy, wbits, kvbits, tolerance)
    tolerant_intersection = tolerant_target & tolerant_proxy

    rng = np.random.default_rng(0)
    null_jaccard = []
    for _ in range(5):
        perturbed = target + rng.uniform(-tolerance, tolerance, len(target))
        null_jaccard.append(
            jaccard(
                tolerant_target,
                pareto_front(perturbed, wbits, kvbits, tolerance),
            )
        )

    cells = load_narrow_cells()
    taus, rhos, hits, regrets = [], [], [], []
    for cell in cells:
        cell_proxy = np.array([pair[0] for pair in cell])
        cell_target = np.array([pair[1] for pair in cell])
        tau = kendalltau(cell_proxy, cell_target).correlation
        rho = spearmanr(cell_proxy, cell_target).correlation
        if np.isfinite(tau) and np.isfinite(rho):
            taus.append(tau)
            rhos.append(rho)
            proxy_best = int(np.argmin(cell_proxy))
            target_best = int(np.argmin(cell_target))
            hits.append(proxy_best == target_best)
            regrets.append(
                100
                * (cell_target[proxy_best] - cell_target[target_best])
                / cell_target[target_best]
            )

    budget_cells = {}
    for i, rounded_wbits in enumerate(np.round(wbits, 1)):
        budget_cells.setdefault(rounded_wbits, []).append(i)
    supply_recall, kept_fraction = [], []
    for indices in budget_cells.values():
        if len(indices) < 20:
            continue
        indices = np.array(indices)
        band = indices[proxy[indices] <= 1.10 * proxy[indices].min()]
        supply_recall.append(indices[np.argmin(target[indices])] in set(band.tolist()))
        kept_fraction.append(len(band) / len(indices))

    print(f"paired production archive: {len(target)} configurations")
    print(
        "exact fronts: "
        f"target={len(exact_target)} proxy={len(exact_proxy)} "
        f"intersection={len(exact_intersection)} "
        f"target_recall={len(exact_intersection) / len(exact_target):.4f} "
        f"jaccard={jaccard(exact_target, exact_proxy):.4f} "
        f"excluded={len(exact_target - exact_proxy)} "
        f"strict_certificates={certificate_count}"
    )
    print(
        f"tolerant fronts (JSD tolerance={tolerance:g}): "
        f"target={len(tolerant_target)} proxy={len(tolerant_proxy)} "
        f"intersection={len(tolerant_intersection)} "
        f"target_recall={len(tolerant_intersection) / len(tolerant_target):.4f} "
        f"jaccard={jaccard(tolerant_target, tolerant_proxy):.4f}"
    )
    print(
        "AWQ noise-null Jaccard: "
        f"{np.mean(null_jaccard):.4f} +/- {np.std(null_jaccard):.4f}"
    )
    print(
        f"narrow cells: n={len(taus)} median_tau={np.median(taus):.4f} "
        f"median_rho={np.median(rhos):.4f} top1_match={100 * np.mean(hits):.2f}% "
        f"regret_median={np.median(regrets):.2f}% "
        f"regret_p90={np.percentile(regrets, 90):.2f}% "
        f"regret_max={np.max(regrets):.2f}%"
    )
    print(
        "stage-1 supply at epsilon=0.10: "
        f"target-best recall={100 * np.mean(supply_recall):.2f}% "
        f"pool kept={100 * np.mean(kept_fraction):.2f}%"
    )


if __name__ == "__main__":
    sys.exit(main())
