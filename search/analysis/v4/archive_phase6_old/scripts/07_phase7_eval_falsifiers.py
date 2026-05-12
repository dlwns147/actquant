"""07_phase7_eval_falsifiers.py — GPU evaluation of acquired falsifier candidates.

Inputs:
    acquired_falsifiers_{tag}.json   (output of 06_phase6_acquire_falsifiers.py)

Outputs (per model):
    eval_falsifiers_{tag}.json    (per-arch: predicted + actual + violation flags)
    eval_falsifiers_{tag}.csv     (flat table)

Run (parallel):
    cd /NAS/SJ/actquant/search
    CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=12345 \
        analysis/v4/07_phase7_eval_falsifiers.py --tag llama --gpu_id 2
    CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=12346 \
        analysis/v4/07_phase7_eval_falsifiers.py --tag qwen  --gpu_id 3

Resumes automatically (re-run after interruption to continue).
"""
import sys, os, json, csv, time, argparse, warnings, gc
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import torch
from evaluator import LlamaEvaluator
from utils.func import init_accelerator, get_net_info, clean_up, set_seed

BASE = '/NAS/SJ/actquant/search'
OUT  = f'{BASE}/analysis/v4'

# Match 260510 AWQ + KIVI config (same as RS50 / RS200 train/test datasets)
DEFAULTS_LLAMA = dict(
    model_path='/SSD/huggingface/meta-llama',
    model_name='Llama-3.1-8B-Instruct',
    config=f'{BASE}/config/llama.json',
    config_key='Llama-3.1-8B-Instruct',
)
DEFAULTS_QWEN = dict(
    model_path='/SSD/huggingface/Qwen',
    model_name='Qwen2.5-7B-Instruct',
    config=f'{BASE}/config/llama.json',  # config supports Qwen via key lookup
    config_key='Qwen2.5-7B-Instruct',
)

EVAL_DEFAULTS = dict(
    dtype='float16',
    w_method=['awq'],
    kv_method='kivi',
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


def load_existing(path):
    if not os.path.exists(path): return []
    with open(path) as f: return json.load(f).get('results', [])


def save_results(path, results, meta=None):
    payload = {'results': results}
    if meta is not None: payload['meta'] = meta
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', choices=['llama', 'qwen'], required=True)
    ap.add_argument('--gpu_id', type=str, required=True)
    ap.add_argument('--max_n', type=int, default=None,
                    help='stop after N new evals (useful for incremental run)')
    args = ap.parse_args()
    set_seed(0)

    cfg = DEFAULTS_LLAMA if args.tag == 'llama' else DEFAULTS_QWEN
    acq_path = f'{OUT}/acquired_falsifiers_{args.tag}.json'
    out_json = f'{OUT}/eval_falsifiers_{args.tag}.json'
    out_csv  = f'{OUT}/eval_falsifiers_{args.tag}.csv'

    if not os.path.exists(acq_path):
        print(f"ERROR: acquisition file missing: {acq_path}\n"
              f"Run analysis/v4/06_phase6_acquire_falsifiers.py first.")
        sys.exit(1)

    with open(acq_path) as f:
        acq = json.load(f)

    records = acq['candidates']
    pf_points = acq['baseline_pf_points']  # 4D: (naive_add, wbits, kvbits, kvdim)
    sigma_resid = acq['sigma_resid_rbf']
    n_total = len(records)
    print(f"[{args.tag}] loaded {n_total} candidates from {acq_path}", flush=True)
    print(f"  σ_resid_rbf={sigma_resid:.5f}  PF size={len(pf_points)}", flush=True)

    # Sort by 3D complexity bucket round-robin so partial runs cover the
    # (wbits, kvbits, kvdim) space early.  Never collapse to 1D total_c.
    NB_W, NB_KV, NB_KD = 3, 2, 2
    wb_arr = np.array([r['wbits']  for r in records])
    kb_arr = np.array([r['kvbits'] for r in records])
    kd_arr = np.array([r['kvdim']  for r in records])
    def _bin(a, nb):
        lo, hi = a.min() - 1e-9, a.max() + 1e-9
        return np.minimum(nb - 1, ((a - lo) / (hi - lo) * nb).astype(int))
    bw = _bin(wb_arr, NB_W); bk = _bin(kb_arr, NB_KV); bd = _bin(kd_arr, NB_KD)
    nb_total = NB_W * NB_KV * NB_KD
    buckets = [[] for _ in range(nb_total)]
    for i in range(len(records)):
        buckets[(bw[i] * NB_KV + bk[i]) * NB_KD + bd[i]].append(i)
    for b in buckets:
        b.sort(key=lambda i: -records[i]['prob_falsifier'])
    order = []
    for round_idx in range(max(len(b) for b in buckets) if buckets else 0):
        for b in buckets:
            if round_idx < len(b): order.append(b[round_idx])
    records = [records[i] for i in order]

    # Resume
    existing = load_existing(out_json)
    done_keys = {(r['w_pool_idx'], r['kv_pool_idx'], r['kvd_pool_idx']) for r in existing}
    print(f"[{args.tag}] resume: {len(existing)} already evaluated, "
          f"{n_total - len(existing)} remaining")

    # Init evaluator
    with open(cfg['config']) as f:
        config = json.load(f)[cfg['config_key']]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(f"[{args.tag}] device_map ready (gpu_id={args.gpu_id})")

    model_id = f"{cfg['model_path']}/{cfg['model_name']}"
    group_size = {'w': EVAL_DEFAULTS['w_group_size'],
                  'k': EVAL_DEFAULTS['k_group_size'],
                  'v': EVAL_DEFAULTS['v_group_size']}

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=model_id,
        method={'w': EVAL_DEFAULTS['w_method'], 'kv': EVAL_DEFAULTS['kv_method']},
        quant_model_paths=[],
        outlier=None,
        seqlen=EVAL_DEFAULTS['seqlen'],
        min_seqlen=EVAL_DEFAULTS['min_seqlen'],
        n_sample=EVAL_DEFAULTS['n_sample'],
        datasets=EVAL_DEFAULTS['datasets'],
        device_map=device_map,
        dtype=EVAL_DEFAULTS['dtype'],
        bits={'w': EVAL_DEFAULTS['w_bits'],
              'k': EVAL_DEFAULTS['k_bits'],
              'v': EVAL_DEFAULTS['v_bits']},
        group_size=group_size,
        residual_length=EVAL_DEFAULTS['residual_length'],
        k_quant_scheme=EVAL_DEFAULTS['k_quant_scheme'],
        v_quant_scheme=EVAL_DEFAULTS['v_quant_scheme'],
        loss_func=EVAL_DEFAULTS['loss_func'],
    )
    accelerator.print(f"[{args.tag}] evaluator ready")

    results = list(existing)
    t_start = time.time()
    n_new = 0
    for i, rec in enumerate(records):
        key = (rec['w_pool_idx'], rec['kv_pool_idx'], rec['kvd_pool_idx'])
        if key in done_keys:
            continue
        if args.max_n is not None and n_new >= args.max_n:
            accelerator.print(f"[max_n={args.max_n}] stop")
            break
        n_new += 1
        t0 = time.time()
        accelerator.print(
            f"\n[{args.tag}][{i+1}/{n_total}] "
            f"pool=({rec['w_pool_idx']},{rec['kv_pool_idx']},{rec['kvd_pool_idx']})  "
            f"wb={rec['wbits']:.3f} kvb={rec['kvbits']:.3f} kvd={rec['kvdim']:.1f}  "
            f"naive_add={rec['naive_add']:.4f}  RBF_μ={rec['rbf_mu']:.4f}  "
            f"P(falsifier)={rec['prob_falsifier']:.3f}")
        try:
            model = evaluator.sample(rec['arch'])
            # Static eval mode matching the RS50/RS200 measurement protocol
            model.config.quant_kv_output = True
            model.config.use_cache = False
            if 'kivi' in EVAL_DEFAULTS['kv_method']:
                model.config.kivi_config.residual_length = 0

            metric = evaluator.eval(arch=rec['arch'], metric=EVAL_DEFAULTS['metric'],
                                    model=model, accelerator=accelerator,
                                    loss_func=EVAL_DEFAULTS['loss_func'],
                                    stride=None)[0]
            y_actual = float(list(metric.values())[0])
            info = get_net_info(rec['arch'], config, group_size)
            actual_eff_kv = float(info.get('eff_kvbits',
                                            info.get('kvbits', 0)))
            actual_total_c = float(info['wbits']) + actual_eff_kv

            res = dict(rec)
            res['y_actual']        = y_actual
            res['actual_wbits']    = float(info['wbits'])
            res['actual_kvbits']   = float(info['kvbits'])
            res['actual_kvdim']    = float(info['kvdim'])
            res['actual_eff_kvbits'] = actual_eff_kv
            res['residual_actual_minus_mu']   = y_actual - rec['rbf_mu']
            res['residual_actual_minus_naive'] = y_actual - rec['naive_add']
            res['eval_time_sec'] = time.time() - t0
            results.append(res)

            accelerator.print(
                f"   y_actual={y_actual:.5f}  residual_vs_μ={y_actual - rec['rbf_mu']:+.5f}  "
                f"residual_vs_naive={y_actual - rec['naive_add']:+.5f}  "
                f"t={res['eval_time_sec']:.1f}s", flush=True)

            if accelerator.is_main_process:
                save_results(out_json, results,
                             meta=dict(model=cfg['model_name'],
                                       sigma_resid_rbf=sigma_resid,
                                       baseline_pf_points=pf_points))
                _write_csv(out_csv, results)

            if 'awq' in EVAL_DEFAULTS['w_method'] or 'gptq' in EVAL_DEFAULTS['w_method']:
                del model
                if hasattr(evaluator, 'model'):
                    del evaluator.model
                clean_up()

        except Exception as e:
            accelerator.print(f"  ERROR on idx {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    t_total = time.time() - t_start
    accelerator.print(f"\n[{args.tag}] done. {n_new} new evals in {t_total/60:.1f} min "
                      f"({t_total/max(n_new,1):.1f} s/arch)")


def _write_csv(path, results):
    if not results: return
    keys = sorted({k for r in results for k in r.keys() if k != 'arch'})
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, '') for k in keys])


if __name__ == '__main__':
    main()
