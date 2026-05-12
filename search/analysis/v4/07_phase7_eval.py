"""07_phase7_eval.py — GPU evaluation of Phase-6 acquired candidates.

Input:  acquired_falsifiers_{tag}.json   (Phase 6 output)
Output: eval_falsifiers_{tag}.{json,csv}  (per-arch measured y_actual + metadata)

Resume-safe: re-running picks up where the previous run left off.

Launch (parallel):
    cd /NAS/SJ/actquant/search
    CUDA_VISIBLE_DEVICES=2 python -u analysis/v4/07_phase7_eval.py --tag llama --gpu_id 2 \
        > analysis/v4/phase7_llama_run.log 2>&1
    CUDA_VISIBLE_DEVICES=3 python -u analysis/v4/07_phase7_eval.py --tag qwen --gpu_id 3 \
        > analysis/v4/phase7_qwen_run.log 2>&1

Measurement protocol matches 260510 RS50/RS200:
    W=AWQ, KV=KIVI, wikitext2 JSD, seqlen=2048, n_sample=128, deterministic seed.
"""
import sys, os, json, csv, time, argparse, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import torch
from evaluator import LlamaEvaluator
from utils.func import init_accelerator, get_net_info, clean_up, set_seed

BASE = '/NAS/SJ/actquant/search'
OUT  = f'{BASE}/analysis/v4'

DEFAULTS = dict(
    llama=dict(model_path='/SSD/huggingface/meta-llama',
               model_name='Llama-3.1-8B-Instruct',
               config=f'{BASE}/config/llama.json'),
    qwen =dict(model_path='/SSD/huggingface/Qwen',
               model_name='Qwen2.5-7B-Instruct',
               config=f'{BASE}/config/qwen2.json'),
)

EVAL_DEFAULTS = dict(
    dtype='float16',
    w_method=['awq'], kv_method='kivi',
    w_bits=[2, 3, 4], k_bits=[2, 4], v_bits=[2, 4],
    w_group_size=128, k_group_size=[128, 128], v_group_size=[128, 128],
    residual_length=128, k_quant_scheme='channel', v_quant_scheme='token',
    metric='loss', loss_func='jsd', datasets=['wikitext2'],
    seqlen=2048, min_seqlen=2048, n_sample=128, data_batch_size=1, seed=0,
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


def write_csv(path, results):
    if not results: return
    keys = sorted({k for r in results for k in r if k != 'arch'})
    with open(path, 'w') as f:
        w = csv.writer(f); w.writerow(keys)
        for r in results: w.writerow([r.get(k, '') for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', choices=['llama', 'qwen'], required=True)
    ap.add_argument('--gpu_id', type=str, required=True)
    ap.add_argument('--max_n', type=int, default=None)
    args = ap.parse_args()
    set_seed(0)

    cfg = DEFAULTS[args.tag]
    acq_path = f'{OUT}/acquired_falsifiers_{args.tag}.json'
    out_json = f'{OUT}/eval_falsifiers_{args.tag}.json'
    out_csv  = f'{OUT}/eval_falsifiers_{args.tag}.csv'

    if not os.path.exists(acq_path):
        print(f"ERROR: missing acquisition file {acq_path}\n"
              f"Run analysis/v4/06_phase6_acquisition.py first.")
        sys.exit(1)
    with open(acq_path) as f:
        acq = json.load(f)
    records = acq['candidates']
    n_total = len(records)
    σ_conf = acq['sigma_conf']
    eps_p  = acq['eps_primary']
    print(f"[{args.tag}] loaded {n_total} candidates  σ_conf={σ_conf:.5f}  ε={eps_p}", flush=True)
    bc = acq.get('bucket_counts', {})
    if bc: print(f"  bucket counts: {bc}", flush=True)

    # Round-robin by bucket for early coverage
    by_bucket = {}
    for r in records:
        by_bucket.setdefault(r['bucket'], []).append(r)
    # within each bucket sort by EVI desc
    for b in by_bucket: by_bucket[b].sort(key=lambda x: -x['EVI_eps0p005'])
    ordered = []
    while any(by_bucket.values()):
        for b in list(by_bucket):
            if by_bucket[b]: ordered.append(by_bucket[b].pop(0))

    existing = load_existing(out_json)
    done_keys = {r['pool_idx'] for r in existing}
    print(f"  resume: {len(existing)} done, {n_total - len(existing)} remaining", flush=True)

    with open(cfg['config']) as f:
        config = json.load(f)[cfg['model_name']]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(f"[{args.tag}] device_map ready (gpu_id={args.gpu_id})")

    model_id = f"{cfg['model_path']}/{cfg['model_name']}"
    group_size = {'w': EVAL_DEFAULTS['w_group_size'],
                  'k': EVAL_DEFAULTS['k_group_size'],
                  'v': EVAL_DEFAULTS['v_group_size']}
    evaluator = LlamaEvaluator(
        config, accelerator=accelerator, model_id=model_id,
        method={'w': EVAL_DEFAULTS['w_method'], 'kv': EVAL_DEFAULTS['kv_method']},
        quant_model_paths=[], outlier=None,
        seqlen=EVAL_DEFAULTS['seqlen'], min_seqlen=EVAL_DEFAULTS['min_seqlen'],
        n_sample=EVAL_DEFAULTS['n_sample'], datasets=EVAL_DEFAULTS['datasets'],
        device_map=device_map, dtype=EVAL_DEFAULTS['dtype'],
        bits={'w': EVAL_DEFAULTS['w_bits'], 'k': EVAL_DEFAULTS['k_bits'], 'v': EVAL_DEFAULTS['v_bits']},
        group_size=group_size, residual_length=EVAL_DEFAULTS['residual_length'],
        k_quant_scheme=EVAL_DEFAULTS['k_quant_scheme'],
        v_quant_scheme=EVAL_DEFAULTS['v_quant_scheme'],
        loss_func=EVAL_DEFAULTS['loss_func'],
    )
    accelerator.print(f"[{args.tag}] evaluator ready")

    results = list(existing)
    t_start = time.time(); n_new = 0
    for i, rec in enumerate(ordered):
        if rec['pool_idx'] in done_keys: continue
        if args.max_n is not None and n_new >= args.max_n:
            accelerator.print(f"[max_n={args.max_n}] stop"); break
        n_new += 1; t0 = time.time()
        accelerator.print(
            f"\n[{args.tag}][{i+1}/{n_total}] {rec['bucket']}  "
            f"pool=({rec['w_pool_idx']},{rec['kv_pool_idx']},{rec['kvd_pool_idx']})  "
            f"layer=({rec['w_layer']},{rec['kv_layer']},{rec['kvd_layer']})  "
            f"wb={rec['wbits']:.3f} kvb={rec['kvbits']:.3f} kvd={rec['kvdim']:.1f}  "
            f"μ={rec['rbf_mu']:.4f} fC*={rec['fC_star_3D']:.4f} "
            f"EVI={rec['EVI_eps0p005']:+.5f} P_ε={rec['P_eps0p005']:.3f}", flush=True)
        try:
            model = evaluator.sample(rec['arch'])
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
            res = dict(rec)
            res['y_actual']        = y_actual
            res['actual_wbits']    = float(info['wbits'])
            res['actual_kvbits']   = float(info['kvbits'])
            res['actual_kvdim']    = float(info['kvdim'])
            res['actual_eff_kvbits'] = float(info.get('eff_kvbits', 0))
            res['residual_actual_minus_mu']    = y_actual - rec['rbf_mu']
            res['residual_actual_minus_fC_3D'] = y_actual - rec['fC_star_3D']
            res['violation_eps0p005'] = bool(res['residual_actual_minus_fC_3D'] < -eps_p)
            res['eval_time_sec'] = time.time() - t0
            results.append(res)

            accelerator.print(
                f"   y={y_actual:.5f}  r_3D={res['residual_actual_minus_fC_3D']:+.5f}  "
                f"violator(ε=0.005)={res['violation_eps0p005']}  "
                f"t={res['eval_time_sec']:.1f}s", flush=True)

            if accelerator.is_main_process:
                save_results(out_json, results,
                             meta=dict(model=cfg['model_name'],
                                       sigma_conf=σ_conf, eps_primary=eps_p,
                                       baseline_pf_points=acq['baseline_pf_points'],
                                       baseline_pf_obj_columns=acq['baseline_pf_obj_columns'],
                                       archive_slack_delta_3=acq['archive_slack_delta_3']))
                write_csv(out_csv, results)
            if 'awq' in EVAL_DEFAULTS['w_method']:
                del model
                if hasattr(evaluator, 'model'): del evaluator.model
                clean_up()
        except Exception as e:
            accelerator.print(f"  ERROR idx {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    t_total = time.time() - t_start
    accelerator.print(f"\n[{args.tag}] done. {n_new} new in {t_total/60:.1f} min "
                      f"({t_total/max(n_new,1):.1f} s/arch)")


if __name__ == '__main__':
    main()
