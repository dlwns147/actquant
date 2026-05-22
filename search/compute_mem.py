import argparse
import csv
import json
import os
from itertools import product
from utils.func import get_net_info

def pair_bits_gs(bits_list, gs_list, name):
    """Match each bit-width with its group size positionally.

    A length-1 group-size list is broadcast to every bit-width; otherwise the
    two lists must have equal length (one-to-one pairing)."""
    if len(gs_list) == 1:
        gs_list = gs_list * len(bits_list)
    if len(gs_list) != len(bits_list):
        raise SystemExit(
            f"--{name}_group_size has {len(gs_list)} values but --{name}_bits "
            f"has {len(bits_list)}; pass 1 (broadcast) or exactly "
            f"{len(bits_list)} group sizes.")
    return list(zip(bits_list, gs_list))

def compute_one(args, config, n_block, w_bits, w_group_size, k_bits, v_bits,
                k_group_size, v_group_size, k_prune_dim, v_prune_dim, n_token):
    arch = {
        'q': {
            'w': {l: [w_bits] * n_block for l in config['linear']},
            'k': [[k_bits, k_group_size]] * n_block,
            'v': [[v_bits, v_group_size]] * n_block,
        },
        'p': {
            'k': [k_prune_dim] * n_block,
            'v': [v_prune_dim] * n_block,
        }
    }
    group_size = {'w': w_group_size, 'k': k_group_size, 'v': v_group_size}
    complexity = get_net_info(arch, config, group_size, n_token,
                              residual_length=args.residual_length)
    return arch, complexity

def normalize_kv_args(args):
    """--kv_bits/--kv_group_size/--kv_prune_dim apply one unified setting to
    both K and V. Any of them sets the matching k_*/v_* lists; returns True
    when KV should be treated as a single axis (V mirrors K)."""
    if args.kv_bits is not None:
        args.k_bits = args.v_bits = args.kv_bits
    if args.kv_group_size is not None:
        args.k_group_size = args.v_group_size = args.kv_group_size
    if args.kv_prune_dim is not None:
        args.k_prune_dim = args.v_prune_dim = args.kv_prune_dim
    return any(v is not None for v in
               (args.kv_bits, args.kv_group_size, args.kv_prune_dim))

def compute_mem(args):
    unify_kv = normalize_kv_args(args)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    n_block = config['n_block']

    # Group sizes pair one-to-one with their bit lists (broadcast if length 1);
    # everything else (including prune dim) takes the cartesian product so a
    # single invocation replaces the bash nested loops.
    w_pairs = pair_bits_gs(args.w_bits, args.w_group_size, 'w')
    k_pairs = pair_bits_gs(args.k_bits, args.k_group_size, 'k')

    # n_token is the first (outermost) product axis so it varies slowest.
    if unify_kv:
        # Unify K and V into one KV axis: V mirrors K exactly (same bits,
        # group size and pruning dim). kv_bits<->kv_group_size pair one-to-one
        # (k_pairs, via pair_bits_gs); kv_prune_dim stays an INDEPENDENT axis
        # taking the cartesian product with k_pairs.
        combos = (
            (wp, kp, kp, kd, kd, nt)
            for nt, wp, kp, kd in product(
                args.n_token, w_pairs, k_pairs, args.k_prune_dim)
        )
    else:
        v_pairs = pair_bits_gs(args.v_bits, args.v_group_size, 'v')
        combos = (
            (wp, kp, vp, kd, vd, nt)
            for nt, wp, kp, vp, kd, vd in product(
                args.n_token, w_pairs, k_pairs, v_pairs,
                args.k_prune_dim, args.v_prune_dim)
        )

    results = []
    for (w_bits, w_gs), (k_bits, k_gs), (v_bits, v_gs), k_prune_dim, v_prune_dim, n_token in combos:
        _, complexity = compute_one(
            args, config, n_block, w_bits, w_gs, k_bits, v_bits,
            k_gs, v_gs, k_prune_dim, v_prune_dim, n_token)
        mem = complexity['memory']

        cfg = {
            'w_bits': w_bits, 'w_group_size': w_gs,
            'k_bits': k_bits, 'k_group_size': k_gs,
            'v_bits': v_bits, 'v_group_size': v_gs,
            'k_prune_dim': k_prune_dim, 'v_prune_dim': v_prune_dim,
            'n_token': n_token,
        }
        results.append({'config': cfg, 'memory': mem, 'complexity': complexity})

    print(f'{"summary":-^80}')
    print(f'model: {args.model_path}/{args.model_name} | {len(results)} combinations')
    for r in results:
        m = r['memory']
        print(f'  {r["config"]} | MEM: {m} B = {(m / (1024 * 1024)):.3f} MB '
              f'= {(m / (1024 * 1024 * 1024)):.3f} GB')

    mem_list = [r['memory'] for r in results]
    print(f'memory list (B): {mem_list}')
    print(f'memory list (MB): {[round(m / (1024 * 1024), 3) for m in mem_list]}')
    print(f'memory list (GB): {[round(m / (1024 * 1024 * 1024), 3) for m in mem_list]}')

    if args.csv_file:
        write_csv(args, results)

    return results


def write_csv(args, results):
    """Horizontal layout: a header row of field names (config keys + every
    get_net_info complexity key), then one row per combination. `memory` is
    reported in bytes only."""
    cfg_keys = list(results[0]['config'].keys())
    # complexity keys minus those already covered by the config rows
    comp_keys = [k for k in results[0]['complexity'].keys()
                 if k not in cfg_keys]
    header = cfg_keys + comp_keys

    path = args.csv_file
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, args.csv_file)

    # results are ordered with n_token outermost; insert one blank row
    # between consecutive n_token groups so each block is visually separated.
    sep_after = {i for i in range(len(results) - 1)
                 if results[i]['config']['n_token']
                 != results[i + 1]['config']['n_token']}

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, r in enumerate(results):
            row = [r['config'][k] for k in cfg_keys] + \
                  [r['complexity'][k] for k in comp_keys]
            writer.writerow(row)
            if i in sep_after:
                writer.writerow([])
    print(f'wrote {len(results)} combinations x '
          f'{len(header)} columns -> {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--w_bits', type=int, nargs='+', default=[4],
                        help='one or more weight bit-widths')
    parser.add_argument('--k_bits', type=int, nargs='+', default=[4],
                        help='one or more key-cache bit-widths')
    parser.add_argument('--v_bits', type=int, nargs='+', default=[4],
                        help='one or more value-cache bit-widths')
    parser.add_argument('--k_prune_dim', type=int, nargs='+', default=[0],
                        help='one or more per-layer K cache prune dims '
                             '(# head_dim channels removed; 0 = no pruning)')
    parser.add_argument('--v_prune_dim', type=int, nargs='+', default=[0],
                        help='one or more per-layer V cache prune dims '
                             '(# head_dim channels removed; 0 = no pruning)')
    parser.add_argument('--w_group_size', type=int, nargs='+', default=[128],
                        help='one or more weight group sizes, paired 1:1 with --w_bits')
    parser.add_argument('--k_group_size', type=int, nargs='+', default=[128],
                        help='one or more key-cache group sizes, paired 1:1 with --k_bits')
    parser.add_argument('--v_group_size', type=int, nargs='+', default=[128],
                        help='one or more value-cache group sizes, paired 1:1 with --v_bits')
    parser.add_argument('--kv_bits', type=int, nargs='+', default=None,
                        help='unified KV bit-widths; sets both --k_bits and '
                             '--v_bits and treats KV as a single axis')
    parser.add_argument('--kv_group_size', type=int, nargs='+', default=None,
                        help='unified KV group sizes, paired 1:1 with --kv_bits; '
                             'sets both --k_group_size and --v_group_size')
    parser.add_argument('--kv_prune_dim', type=int, nargs='+', default=None,
                        help='unified KV prune dims (# head_dim channels '
                             'removed; 0 = no pruning); sets both '
                             '--k_prune_dim and --v_prune_dim. Independent '
                             'axis (cartesian product), not paired with '
                             '--kv_bits')
    parser.add_argument('--residual_length', type=int, default=128,
                        help='')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--n_token', type=int, nargs='+', default=[0],
                        help='one or more target sequence lengths for memory calculation')
    parser.add_argument('--save', type=str, default='',
                        help='directory to write --csv_file into (created if missing)')
    parser.add_argument('--csv_file', type=str, default='',
                        help='if set, save every combination + complexity to this CSV')

    cfgs = parser.parse_args()
    compute_mem(cfgs)
