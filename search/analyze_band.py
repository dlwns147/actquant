"""Standalone: how many joint-archive archs fall inside the memory comp_obj
band, as a function of (memory budget, threshold). Mirrors post_search.select_joint
+ utils.func.compute_memory (no torch needed)."""
import json, sys
import numpy as np

CFG = json.load(open('config/llama.json'))['Llama-3.1-8B-Instruct']
GROUP_SIZE = {'w': 128, 'k': 128, 'v': 128}
THINK_RESIDUAL = 32


def compute_memory(arch, config, group_size, n_token=0, residual_length=0,
                   think_residual=THINK_RESIDUAL, sink=0):
    w_group_size = group_size['w']
    weight_memory = 0
    for linear in config['linear']:
        out_dim, in_dim = map(int, config['linear_shape'][linear])
        lgs = in_dim if w_group_size == -1 else w_group_size
        linear_bits = arch['q']['w'][linear]
        if isinstance(linear_bits[0], int):
            for bits in linear_bits:
                weight_memory += out_dim * in_dim * bits // 8
                if bits < 16:
                    weight_memory += (in_dim // lgs) * out_dim * 4
        else:
            for bits, n_out in linear_bits:
                weight_memory += out_dim * in_dim * bits // 8
                if bits < 16:
                    weight_memory += (in_dim // lgs) * out_dim * 4
                weight_memory += out_dim * n_out * 2
    weight_memory += int(config['vocab_size']) * int(config['hidden_size']) * 4
    weight_memory += int(config['n_norm']) * int(config['hidden_size']) * 2
    weight_memory += int(config['max_position_embeddings']) * int(config['head_dim']) * 2

    head_dim = int(config['head_dim'])
    p_arch = arch.get('p', {})
    R = max(int(residual_length), 0) if residual_length else 0
    S = max(int(sink), 0)
    cache_memory = 0
    for target in ['k', 'v']:
        kv_dim = int(config['linear_shape'][config[f'{target}_linear']][0])
        n_kv_heads = kv_dim // head_dim
        prune_list = p_arch.get(target, [0] * len(arch['q'][target]))
        for (bits, gs), prune_dim in zip(arch['q'][target], prune_list):
            n = int(n_token); r = min(R, n); s = min(S, n)
            t = min(int(think_residual), n) if prune_dim > 0 else n
            c = {('fp','full'):0,('fp','pru'):0,('q','full'):0,('q','pru'):0}
            bps = sorted({0, s, n - r, n - t, n})
            for a, b in zip(bps, bps[1:]):
                if b <= a:
                    continue
                is_fp = (a < s) or (a >= n - r)
                is_full = (a < s) or (a >= n - t)
                c[('fp' if is_fp else 'q', 'full' if is_full else 'pru')] += b - a
            full_dim = n_kv_heads * head_dim
            pruned_dim = n_kv_heads * (head_dim - prune_dim)
            def _q_mem(dim):
                m = dim * bits / 8
                if bits < 16:
                    m += (dim / gs) * 4
                return m
            cache_memory += c[('fp','full')] * full_dim * 16 / 8
            cache_memory += c[('fp','pru')] * pruned_dim * 16 / 8
            cache_memory += c[('q','full')] * _q_mem(full_dim)
            cache_memory += c[('q','pru')] * _q_mem(pruned_dim)
    return weight_memory + cache_memory


STATS = ('save/second_search/2606231052_Llama-3.1-8B-Instruct_joint_hqq_kivi_'
         'think_rbf_doe500_it200n50_sk8_s0/iter_200.stats')
N_TOKEN = 16384
ATTN_SINK = 8

sf = json.load(open(STATS))
archive = sf['archive']
archs = [e[0] for e in archive]
loss = np.array([float(e[1]) for e in archive], float)
print(f'archive size = {len(archs)}')

mem = np.array([compute_memory(a, CFG, GROUP_SIZE, n_token=N_TOKEN,
                               residual_length=0, sink=ATTN_SINK) for a in archs])
print(f'memory range over archive: [{mem.min():.4e}, {mem.max():.4e}]  '
      f'median {np.median(mem):.4e}')

# 16384-token budget list from sbatch/actquant/iter/post_search.sh
BUDGETS = [4957511680, 5074591744, 5103165440, 5307637760, 5170110464,
           5287190528, 5315764224, 5520236544, 5829926912, 5947006976,
           5975580672, 6180052992]
THRESHOLDS = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]

print('\nin-box arch count  (rows = memory budget, cols = threshold)')
hdr = 'budget(GB)   ' + ''.join(f'{t:>8}' for t in THRESHOLDS)
print(hdr); print('-' * len(hdr))
for val in BUDGETS:
    counts = []
    for thr in THRESHOLDS:
        d = val * thr
        lo, hi = val - d, val + d
        counts.append(int(((mem >= lo) & (mem <= hi)).sum()))
    print(f'{val/1e9:8.3f}     ' + ''.join(f'{c:>8}' for c in counts))

# summary at the production threshold 0.005
print('\n@threshold=0.005 (production): best-JSD arch in each band')
for val in BUDGETS:
    d = val * 0.005
    feas = np.where((mem >= val - d) & (mem <= val + d))[0]
    if len(feas) == 0:
        print(f'{val/1e9:.3f} GB: EMPTY'); continue
    order = feas[np.argsort(loss[feas])]
    print(f'{val/1e9:.3f} GB: n_inbox={len(feas):4d}  '
          f'best JSD={loss[order[0]]:.5f}  worst JSD={loss[order[-1]]:.5f}  '
          f'JSD spread={loss[feas].max()-loss[feas].min():.5f}')

# ── does widening the band actually buy lower JSD, and at what memory cost? ──
# For each budget: best JSD inside band + the memory the best-JSD arch uses,
# expressed as a signed % offset from the budget center. If widening just keeps
# grabbing the TOP of the band (offset → +thr), it's borrowing memory, not skill.
print('\nbest-JSD inside band vs threshold  (J=best JSD, off%=mem offset of that '
      'arch from budget center)')
for val in BUDGETS:
    row = f'{val/1e9:6.3f}GB '
    for thr in THRESHOLDS:
        d = val * thr
        feas = np.where((mem >= val - d) & (mem <= val + d))[0]
        if len(feas) == 0:
            row += f' | {thr}: empty'.ljust(0); row += f'  {thr:>6}:  ----        '
            continue
        b = feas[np.argmin(loss[feas])]
        off = (mem[b] - val) / val * 100
        row += f'  {thr:>6}:J{loss[b]:.3f}@{off:+5.2f}%'
    print(row)

# verify_topk feasibility: how often are >=5 candidates available?
print('\nmin in-box count across all 12 budgets, per threshold '
      '(want >= verify_topk):')
for thr in THRESHOLDS:
    cs = []
    for val in BUDGETS:
        d = val * thr
        cs.append(int(((mem >= val - d) & (mem <= val + d)).sum()))
    print(f'  thr={thr:<7} min={min(cs):4d}  (empty bands: '
          f'{sum(c == 0 for c in cs)})')
