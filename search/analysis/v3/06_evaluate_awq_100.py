"""06_evaluate_awq_100.py — Run AWQ + KIVI on the 100 acquired arch list and
analyse falsification of the Cartesian-combined Pareto front.

Inputs:
  analysis/v3/acquired_offsurface_100.json  (output of 05_acquire_offsurface_100.py)

Outputs:
  analysis/v3/eval_offsurface_100.json    (per-arch: pred + actual + violation)
  analysis/v3/eval_offsurface_100.csv     (flat table for downstream)
  analysis/v3/eval_offsurface_100.txt     (statistical summary, ε bounds, CI)
  figures/06_eval_offsurface_overview.png

Run:
  cd /NAS/SJ/actquant/search
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 \
      --main_process_port=12345 analysis/v3/06_evaluate_awq_100.py --gpu_id 0

Resumes automatically — re-run after interruption to continue.
"""
import sys, os, json, csv, time, argparse, warnings, gc
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluator import LlamaEvaluator
from utils.func import init_accelerator, get_net_info, set_seed, clean_up

# ─── Defaults (match AWQ_3WAY training run) ─────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
ACQ_JSON   = f'{BASE}/analysis/v3/acquired_offsurface_100.json'
OUT_JSON   = f'{BASE}/analysis/v3/eval_offsurface_100.json'
OUT_CSV    = f'{BASE}/analysis/v3/eval_offsurface_100.csv'
OUT_TXT    = f'{BASE}/analysis/v3/eval_offsurface_100.txt'
FIG_PATH   = f'{BASE}/analysis/v3/figures/06_eval_offsurface_overview.png'

# Match AWQ_3WAY config (Llama-3.1-8B-Instruct, AWQ + KIVI, wikitext2, JSD)
DEFAULTS = dict(
    model_path='/SSD/huggingface/meta-llama',
    model_name='Llama-3.1-8B-Instruct',
    config=f'{BASE}/config/llama.json',
    dtype='float16',
    w_method=['awq'],
    kv_method=['kivi'],
    w_bits=[2, 3, 4],
    k_bits=[2, 4],
    v_bits=[2, 4],
    w_group_size=128,
    k_group_size=[128, 128],
    v_group_size=[128, 128],
    residual_length=128,
    k_quant_scheme='channel',
    v_quant_scheme='token',
    metric='loss',
    loss_func='jsd',
    datasets=['wikitext2'],
    seqlen=2048,
    min_seqlen=2048,
    n_sample=128,
    data_batch_size=1,
    seed=0,
)

# ─── Resume support ──────────────────────────────────────────────────────────
def load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get('results', [])

def save_results(path, results, summary=None):
    payload = {'results': results}
    if summary is not None:
        payload['summary'] = summary
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)

# ─── Statistics / figures ───────────────────────────────────────────────────
def rule_of_three_upper(violations, n, alpha=0.05):
    """One-sided upper bound on violation rate at confidence 1−α.
    For 0/n: rule-of-three (3/n at 95%).  For k>0: Clopper-Pearson upper.
    Returns proportion in [0,1]."""
    from scipy.stats import beta as _beta
    if n == 0: return 1.0
    if violations == 0:
        return -np.log(alpha) / n   # Wilks/rule of three (=3/n at α=0.05)
    return float(_beta.ppf(1 - alpha, violations + 1, n - violations))

def compute_summary(results, eps_list=(0.005, 0.01, 0.02, 0.05)):
    if not results:
        return {}
    n = len(results)
    actuals = np.array([r['y_actual'] for r in results])
    preds   = np.array([r['pred_mu']  for r in results])
    pf      = np.array([r['pf_baseline'] for r in results])
    sigma   = np.array([r['pred_sigma'] for r in results])
    probs   = np.array([r['prob_violator'] for r in results])
    total_c = np.array([r['total_c'] for r in results])

    # Residuals
    resid_pred = actuals - preds          # ARD-GP error: y − μ
    resid_pf   = actuals - pf             # vs combined PF baseline
    eps_inf    = float(np.max(np.abs(resid_pred)))
    eps_2      = float(np.sqrt(np.mean(resid_pred**2)))

    # Violations: actual y < f*(c) − ε   (i.e., dominate PF by ε)
    summary = {
        'n': n,
        'actual_loss': {
            'min': float(actuals.min()), 'max': float(actuals.max()),
            'mean': float(actuals.mean()), 'std': float(actuals.std()),
        },
        'ard_gp_residual_actual_minus_pred': {
            'mean': float(resid_pred.mean()), 'std': float(resid_pred.std()),
            'p95_abs': float(np.percentile(np.abs(resid_pred), 95)),
            'eps_inf': eps_inf, 'eps_2': eps_2,
        },
        'pf_residual_actual_minus_baseline': {
            'mean': float(resid_pf.mean()), 'std': float(resid_pf.std()),
            'min': float(resid_pf.min()), 'max': float(resid_pf.max()),
            'p05': float(np.percentile(resid_pf, 5)),
            'p95': float(np.percentile(resid_pf, 95)),
        },
        'violations_vs_combined_pf': {},
        'predicted_vs_actual_violation_calibration': {},
    }
    for eps in eps_list:
        n_viol = int(np.sum(actuals < pf - eps))
        ub = rule_of_three_upper(n_viol, n)
        summary['violations_vs_combined_pf'][f'eps={eps}'] = {
            'count': n_viol, 'rate': n_viol / n,
            'upper_95ci': ub,
        }

    # Predictor calibration: predicted P(violator) vs actual violation rate
    # Bin by predicted P, count actual violations in each bin
    bin_edges = [0, 0.05, 0.10, 0.20, 0.40, 1.0]
    cal = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs >= lo) & (probs < hi)
        n_b = int(in_bin.sum())
        if n_b == 0:
            cal.append({'P_range': f'[{lo},{hi})', 'n': 0, 'actual_violation_rate': None})
        else:
            n_viol_b = int(np.sum((actuals < pf)[in_bin]))
            cal.append({'P_range': f'[{lo},{hi})', 'n': n_b,
                        'actual_violation_rate': n_viol_b / n_b,
                        'mean_pred_P': float(probs[in_bin].mean())})
    summary['predicted_vs_actual_violation_calibration'] = cal

    return summary

def plot_diagnostics(results, fig_path):
    if not results:
        return
    actuals = np.array([r['y_actual'] for r in results])
    preds   = np.array([r['pred_mu']  for r in results])
    pf      = np.array([r['pf_baseline'] for r in results])
    sigma   = np.array([r['pred_sigma'] for r in results])
    probs   = np.array([r['prob_violator'] for r in results])
    total_c = np.array([r['total_c'] for r in results])
    src     = np.array([r['src'] for r in results])
    is_main = src == 'P(violator)'
    is_sig  = src == 'sigma|P>1%'

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) actual vs predicted
    ax = axes[0, 0]
    lo = min(actuals.min(), preds.min()); hi = max(actuals.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, label='y=ŷ')
    sc = ax.scatter(preds, actuals, c=sigma, cmap='viridis', s=28,
                    edgecolor='black', linewidth=0.3)
    plt.colorbar(sc, ax=ax, fraction=0.04, label='σ_θ')
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    ax.set_xlabel('ARD-GP predicted μ'); ax.set_ylabel('AWQ-measured y')
    ax.set_title(f'(a) actual vs predicted  (RMSE={rmse:.4f})', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) actual vs Cartesian-PF baseline (the falsification plot)
    ax = axes[0, 1]
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, label='y=f*(c)')
    for eps, ls_, lbl in [(0.005, ':', 'ε=0.005'), (0.01, '-.', 'ε=0.01'),
                           (0.02, '--', 'ε=0.02')]:
        ax.plot([lo, hi], [lo - eps, hi - eps], 'r', ls=ls_, lw=0.8, label=lbl)
    ax.scatter(pf[is_main], actuals[is_main], s=28, c='red',
               edgecolor='black', linewidth=0.3, label='P(violator)-pick')
    ax.scatter(pf[is_sig],  actuals[is_sig],  s=28, c='orange',
               edgecolor='black', linewidth=0.3, label='σ-extras')
    ax.set_xlabel('Cartesian-PF baseline f*(c)')
    ax.set_ylabel('AWQ-measured y')
    ax.set_title('(b) Falsification: y < f* − ε ⇒ violator', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) residual vs total complexity
    ax = axes[1, 0]
    resid_pf = actuals - pf
    ax.scatter(total_c[is_main], resid_pf[is_main], s=24, c='red',
               edgecolor='black', linewidth=0.3, label='P(violator)-pick')
    ax.scatter(total_c[is_sig],  resid_pf[is_sig],  s=24, c='orange',
               edgecolor='black', linewidth=0.3, label='σ-extras')
    ax.axhline(0, color='black', lw=0.6)
    for eps, c in [(0.005, '#9b59b6'), (0.01, '#c0392b'), (0.02, '#e67e22')]:
        ax.axhline(-eps, color=c, ls='--', lw=0.7, label=f'−ε={eps}')
    ax.set_xlabel('wbits + eff_kvbits')
    ax.set_ylabel('y_actual − f*(c)  (negative ⇒ violator)')
    ax.set_title('(c) Residual vs combined PF baseline', fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # (d) calibration: predicted P vs actual violation rate
    ax = axes[1, 1]
    bin_edges = np.array([0, 0.05, 0.10, 0.20, 0.40, 1.0])
    centers, rates, ns = [], [], []
    for lo_, hi_ in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs >= lo_) & (probs < hi_)
        n_b = int(in_bin.sum())
        if n_b > 0:
            centers.append(probs[in_bin].mean())
            rates.append(np.mean((actuals < pf)[in_bin]))
            ns.append(n_b)
    if centers:
        centers = np.array(centers); rates = np.array(rates); ns = np.array(ns)
        ax.scatter(centers, rates, s=40 + 8 * ns, c='red',
                   edgecolor='black', linewidth=0.3, alpha=0.85,
                   label='binned rate (size ∝ n)')
        for c_, r_, n_ in zip(centers, rates, ns):
            ax.annotate(f'n={n_}', (c_, r_), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, label='ideal calibration')
    ax.set_xlabel('mean predicted P(violator) per bin')
    ax.set_ylabel('actual violation rate (y < f*)')
    ax.set_xlim(-0.02, 0.55); ax.set_ylim(-0.02, 1.02)
    ax.set_title('(d) Calibration: predicted vs actual', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Falsification of Cartesian-combined PF (n={len(results)})',
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()

# ─── Main eval loop ──────────────────────────────────────────────────────────
def main(args):
    set_seed(DEFAULTS['seed'])

    # Load arch list (main + optional low-P controls)
    with open(ACQ_JSON) as f:
        acq = json.load(f)
    archs = list(acq['archs'])
    records = list(acq['records'])
    print(f"[load] {len(archs)} archs from {ACQ_JSON}")
    # Auto-load low-P controls if present
    ctrl_path = f'{BASE}/analysis/v3/acquired_lowP_controls_25.json'
    if os.path.exists(ctrl_path):
        with open(ctrl_path) as f:
            ctrl = json.load(f)
        archs.extend(ctrl['archs'])
        records.extend(ctrl['records'])
        print(f"[load] +{len(ctrl['archs'])} low-P controls from {ctrl_path}")
    assert len(archs) == len(records)
    n_total = len(archs)
    print(f"[load] total: {n_total} archs")

    # Re-order: 2D bucket round-robin (5 wbits × 5 eff_kvbits), top-P within
    # each bucket. Gives diverse 2D-complexity coverage early.
    NB_W, NB_KV = 5, 5
    wb = np.array([r['wbits'] for r in records])
    ek = np.array([r['eff_kvbits'] for r in records])
    w_edges  = np.quantile(wb, np.linspace(0, 1, NB_W  + 1))
    kv_edges = np.quantile(ek, np.linspace(0, 1, NB_KV + 1))
    w_edges[0] -= 1e-9;  w_edges[-1]  += 1e-9
    kv_edges[0] -= 1e-9; kv_edges[-1] += 1e-9
    def bid(cw, ckv):
        bw  = max(0, min(NB_W - 1, int(np.searchsorted(w_edges, cw,  side='right') - 1)))
        bkv = max(0, min(NB_KV - 1, int(np.searchsorted(kv_edges, ckv, side='right') - 1)))
        return bw * NB_KV + bkv
    bucket_lists = [[] for _ in range(NB_W * NB_KV)]
    for i in range(len(records)):
        bucket_lists[bid(wb[i], ek[i])].append(i)
    for b in range(len(bucket_lists)):
        bucket_lists[b].sort(key=lambda i: -records[i]['prob_violator'])
    order = []
    for round_idx in range(max(len(b) for b in bucket_lists) if bucket_lists else 0):
        for b in range(len(bucket_lists)):
            if round_idx < len(bucket_lists[b]):
                order.append(bucket_lists[b][round_idx])
    archs   = [archs[i] for i in order]
    records = [records[i] for i in order]
    print(f"[order] round-robin × {NB_W}×{NB_KV} 2D buckets, top-P within bucket")
    p_str = ", ".join(f"{r['prob_violator']:.3f}" for r in records[:5])
    cw_str = ", ".join(f"{r['wbits']:.2f}" for r in records[:5])
    ce_str = ", ".join(f"{r['eff_kvbits']:.2f}" for r in records[:5])
    print(f"  first 5 P:          [{p_str}]")
    print(f"  first 5 wbits:      [{cw_str}]")
    print(f"  first 5 eff_kvbits: [{ce_str}]")

    # Resume from prior run
    existing = load_existing(OUT_JSON)
    done_indices = {r['sel_idx'] for r in existing}
    print(f"[resume] {len(existing)} already evaluated, "
          f"{n_total - len(existing)} remaining")

    # Init evaluator (heavy: loads tokenizer + dense logits model for JSD)
    with open(DEFAULTS['config']) as f:
        config = json.load(f)[DEFAULTS['model_name']]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(f"device_map ready")

    model_id = f"{DEFAULTS['model_path']}/{DEFAULTS['model_name']}"
    group_size = {'w': DEFAULTS['w_group_size'],
                  'k': DEFAULTS['k_group_size'],
                  'v': DEFAULTS['v_group_size']}

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=model_id,
        method={'w': DEFAULTS['w_method'], 'kv': DEFAULTS['kv_method']},
        quant_model_paths=[],   # AWQ does on-the-fly
        outlier=None,
        seqlen=DEFAULTS['seqlen'],
        min_seqlen=DEFAULTS['min_seqlen'],
        n_sample=DEFAULTS['n_sample'],
        datasets=DEFAULTS['datasets'],
        device_map=device_map,
        dtype=DEFAULTS['dtype'],
        bits={'w': DEFAULTS['w_bits'], 'k': DEFAULTS['k_bits'],
              'v': DEFAULTS['v_bits']},
        group_size=group_size,
        residual_length=DEFAULTS['residual_length'],
        k_quant_scheme=DEFAULTS['k_quant_scheme'],
        v_quant_scheme=DEFAULTS['v_quant_scheme'],
        loss_func=DEFAULTS['loss_func'],
    )
    accelerator.print(f"evaluator ready")

    results = list(existing)
    t_start = time.time()
    n_new = 0
    for i, (arch, rec) in enumerate(zip(archs, records)):
        if rec['sel_idx'] in done_indices:
            continue
        if args.max_n is not None and n_new >= args.max_n:
            accelerator.print(f"[max_n={args.max_n}] stop after {n_new} new evals")
            break
        n_new += 1
        t0 = time.time()
        accelerator.print(f"\n[{i+1}/{n_total}] sel_idx={rec['sel_idx']}, "
                          f"src={rec['src']}, "
                          f"P={rec['prob_violator']:.3f}, "
                          f"pred_μ={rec['pred_mu']:.4f}, "
                          f"f*={rec['pf_baseline']:.4f}, "
                          f"σ={rec['pred_sigma']:.4f}, "
                          f"total_c={rec['total_c']:.3f}")
        try:
            model = evaluator.sample(arch)
            # use_cache=False, quant_kv_output=True for static eval (matches AWQ_3WAY)
            model.config.quant_kv_output = True
            model.config.use_cache = False
            if 'kivi' in DEFAULTS['kv_method']:
                model.config.kivi_config.residual_length = 0

            metric = evaluator.eval(arch=arch, metric=DEFAULTS['metric'],
                                    model=model, accelerator=accelerator,
                                    loss_func=DEFAULTS['loss_func'],
                                    stride=None)[0]
            y_actual = float(list(metric.values())[0])
            info = get_net_info(arch, config, group_size)

            res = dict(rec)
            res['y_actual'] = y_actual
            res['actual_wbits'] = float(info['wbits'])
            res['actual_eff_kvbits'] = float(info.get('eff_kvbits',
                                                      info.get('kvbits', 0)))
            res['actual_total_c'] = float(info['wbits']) + res['actual_eff_kvbits']
            for eps in (0.005, 0.01, 0.02, 0.05):
                res[f'violation_eps_{eps}'] = bool(y_actual < rec['pf_baseline'] - eps)
            res['eval_time_sec'] = time.time() - t0

            results.append(res)
            accelerator.print(f"   y={y_actual:.5f}, "
                              f"residual={y_actual - rec['pf_baseline']:+.5f}, "
                              f"violator(ε=0.01)={res['violation_eps_0.01']}, "
                              f"t={res['eval_time_sec']:.1f}s")

            # Incremental save every arch (AWQ is expensive; keep partials)
            if accelerator.is_main_process:
                summary = compute_summary(results)
                save_results(OUT_JSON, results, summary)

            # Cleanup AWQ model (needs fresh load each arch)
            if 'awq' in DEFAULTS['w_method'] or 'gptq' in DEFAULTS['w_method']:
                del model
                if hasattr(evaluator, 'model'):
                    evaluator.model = None
                clean_up()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            accelerator.print(f"   ERROR: {e}\n{traceback.format_exc()}")
            res = dict(rec); res['error'] = str(e)
            results.append(res)
            if accelerator.is_main_process:
                save_results(OUT_JSON, results)
            clean_up(); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    accelerator.print(f"\n[done] total time: {(time.time() - t_start)/60:.1f} min")
    if not accelerator.is_main_process:
        return

    # ─── Final summary + figure + CSV ───────────────────────────────────────
    valid = [r for r in results if 'y_actual' in r]
    summary = compute_summary(valid)
    save_results(OUT_JSON, results, summary)

    # CSV (flat table)
    if valid:
        cols = ['sel_idx', 'src', 'pred_mu', 'pred_sigma', 'pf_baseline',
                'pred_gap', 'prob_violator', 'wbits', 'kvbits', 'kvdim',
                'eff_kvbits', 'total_c', 'on_cart_pf',
                'y_actual', 'violation_eps_0.005', 'violation_eps_0.01',
                'violation_eps_0.02', 'violation_eps_0.05', 'eval_time_sec']
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in valid: w.writerow(r)
        print(f"saved CSV → {OUT_CSV}")

    # Text summary
    lines = []
    lines.append(f"Falsification of Cartesian-combined Pareto Front")
    lines.append(f"=" * 70)
    lines.append(f"")
    lines.append(f"n_evaluated = {len(valid)} / {n_total}")
    lines.append(f"")
    lines.append(f"AWQ-measured loss y_actual (n={len(valid)}):")
    s = summary['actual_loss']
    lines.append(f"  range: [{s['min']:.5f}, {s['max']:.5f}]   "
                 f"mean: {s['mean']:.5f}   std: {s['std']:.5f}")
    lines.append(f"")
    lines.append(f"ARD-GP residual  (y_actual − pred_μ):")
    s = summary['ard_gp_residual_actual_minus_pred']
    lines.append(f"  mean:    {s['mean']:+.5f}   std: {s['std']:.5f}")
    lines.append(f"  |r|_p95: {s['p95_abs']:.5f}")
    lines.append(f"  ε_∞:     {s['eps_inf']:.5f}   ε_2: {s['eps_2']:.5f}")
    lines.append(f"")
    lines.append(f"PF residual  (y_actual − f*(c)):")
    s = summary['pf_residual_actual_minus_baseline']
    lines.append(f"  range:   [{s['min']:+.5f}, {s['max']:+.5f}]   "
                 f"mean: {s['mean']:+.5f}")
    lines.append(f"  p05:     {s['p05']:+.5f}   p95: {s['p95']:+.5f}")
    lines.append(f"  (negative ⇒ beats Cartesian PF)")
    lines.append(f"")
    lines.append(f"ε-violations (y < f* − ε):")
    for k, v in summary['violations_vs_combined_pf'].items():
        lines.append(f"  {k}:  {v['count']:3d}/{len(valid)}  "
                     f"({100*v['rate']:.1f}%)   "
                     f"95% upper CI on rate: {100*v['upper_95ci']:.2f}%")
    lines.append(f"")
    lines.append(f"Calibration (binned predicted P vs actual rate):")
    for c in summary['predicted_vs_actual_violation_calibration']:
        if c['n'] > 0:
            lines.append(f"  P∈{c['P_range']:>10s}: n={c['n']:3d}  "
                         f"mean_pred={c.get('mean_pred_P', 0):.3f}   "
                         f"actual_rate={c['actual_violation_rate']:.3f}")
        else:
            lines.append(f"  P∈{c['P_range']:>10s}: n=0")

    with open(OUT_TXT, 'w') as f: f.write('\n'.join(lines))
    print('\n' + '\n'.join(lines))

    # Figure
    plot_diagnostics(valid, FIG_PATH)
    print(f"\nsaved figure → {FIG_PATH}")
    print(f"saved JSON   → {OUT_JSON}")
    print(f"saved TXT    → {OUT_TXT}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--max_n', type=int, default=None,
                        help='Stop after evaluating N archs (incremental testing)')
    parser.add_argument('--analyse_only', action='store_true',
                        help='Skip eval, just re-compute summary/figure from existing JSON')
    parser.add_argument('--synthetic_test', action='store_true',
                        help='Generate synthetic y_actual to test analysis pipeline')
    args = parser.parse_args()

    if args.synthetic_test:
        # Pipeline test: synthetic y_actual = pred_μ + ε  (no AWQ needed)
        with open(ACQ_JSON) as f:
            acq = json.load(f)
        rng = np.random.RandomState(0)
        results = []
        for rec in acq['records']:
            y = rec['pred_mu'] + rng.normal(0, rec['pred_sigma'])
            r = dict(rec)
            r['y_actual'] = float(y)
            for eps in (0.005, 0.01, 0.02, 0.05):
                r[f'violation_eps_{eps}'] = bool(y < rec['pf_baseline'] - eps)
            r['eval_time_sec'] = 0.0
            results.append(r)
        out = OUT_JSON.replace('.json', '_synthetic.json')
        fig = FIG_PATH.replace('.png', '_synthetic.png')
        summary = compute_summary(results)
        save_results(out, results, summary)
        plot_diagnostics(results, fig)
        print(f"synthetic test n={len(results)}")
        print(f"  saved {out}\n  saved {fig}")
        for k, v in summary['violations_vs_combined_pf'].items():
            print(f"  {k}: {v['count']}/{len(results)}  CI≤{100*v['upper_95ci']:.2f}%")
    elif args.analyse_only:
        with open(OUT_JSON) as f:
            results = json.load(f).get('results', [])
        valid = [r for r in results if 'y_actual' in r]
        summary = compute_summary(valid)
        save_results(OUT_JSON, results, summary)
        plot_diagnostics(valid, FIG_PATH)
        print(f"re-analysed n={len(valid)}; saved {FIG_PATH}, {OUT_JSON}")
    else:
        main(args)
