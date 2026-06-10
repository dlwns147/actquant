import numpy as np
from utils.func import get_net_info
from tqdm import tqdm
import math


# QEFT keeps `n_outlier` weight columns in FP16; for kernel-friendly packing the
# count is a multiple of 32. When any count > 0 is requested, each weight linear's
# search option becomes a (w_bits, n_outlier) tuple (analogous to KV's (bits, gs)).
def build_qeft_w_options(w_bits, n_qeft_column, eligible_bits, with_outlier=True):
    """De-duplicated, order-preserving [(bits, n_outlier), ...].

    Outlier columns are attached only to `eligible_bits` widths (default: sub-16
    bit, set by caller); 16-bit weights get a single (16, 0). with_outlier=False
    forces the plain (bits, 0) axis (used for o_proj, which QEFT does not OWQ)."""
    opts = []
    for b in w_bits:
        cols = n_qeft_column if (with_outlier and b in eligible_bits) else [0]
        for c in cols:
            if (b, c) not in opts:
                opts.append((b, c))
    return opts


class LlamaSearchSpace:
    """Unified Llama search space (weights + KV group-size + ThinK KV-pruning + QEFT outliers).

    This is the single canonical search space — it subsumes the former
    LlamaSearchSpace / LlamaGroupSizeSearchSpace / LlamaGroupSizeQEFTSearchSpace /
    LlamaThinKSearchSpace. Disabling individual axes recovers each old variant:
      * n_qeft_column=[0]                  -> scalar weight-bit options (no QEFT outliers)
      * k/v_pruning_dim=[0]                -> no ThinK channel pruning
      * single-option K/V group sizes      -> plain per-layer KV bits

    Canonical arch schema (same as awqgptq.py):
        arch = {
            'q': {
                'w': {linear: [w_entry, ...], ...},  # w_entry = bits  OR  [bits, n_outlier]
                'k': [[bits, group_size], ...],       # per-layer K (bits, gs)
                'v': [[bits, group_size], ...],       # per-layer V (bits, gs)
            },
            'p': {
                'k': [pruning_dim, ...],              # per-layer K dims removed from head_dim
                'v': [pruning_dim, ...],              # per-layer V dims removed from head_dim
            }
        }

    QEFT weight option layout is (w_bits, n_outlier) — bit first, then the FP16
    outlier-column count (a multiple of 32). pruning_dim: 0 = no pruning,
    head_dim//2 = prune 50%; used by utils/func.py as
    effective_bits = (bits + scale_overhead) * (1 - pruning_dim / head_dim).

    Encoding layout (flat integer array, length = (n_linear + 4) * n_block):
        [w_indices..., k_indices..., v_indices..., k_dim_indices..., v_dim_indices...]
    """

    def __init__(self,
                 bits,
                 group_size,
                 pass_module,
                 config=None,
                 comp_obj='',
                 comp_obj_min=[],
                 comp_obj_max=[],
                 outlier_bits=[],
                 only_outlier_bits=False,
                 n_token=0,
                 rand_size=5,
                 k_pruning_dim=None,
                 v_pruning_dim=None,
                 n_qeft_column=[0],
                 qeft_outlier_bits=None,
                 ):

        assert 'w' in pass_module and 'k' in pass_module and 'v' in pass_module

        # --- Weight options -------------------------------------------------
        # QEFT outlier-column axis: when any count > 0 is requested, weight options
        # become (w_bits, n_outlier) tuples (o_proj excluded — not OWQ'd). Default
        # n_qeft_column=[0] keeps the legacy scalar-bit options (HQQ/AWQ/GPTQ).
        self.n_qeft_column = list(n_qeft_column)
        self.w_outlier = any(c > 0 for c in self.n_qeft_column)
        if self.w_outlier:
            eligible = set(qeft_outlier_bits) if qeft_outlier_bits else {b for b in bits['w'] if b < 16}
            owq = lambda: build_qeft_w_options(bits['w'], self.n_qeft_column, eligible)
            self.q_proj_option = owq()
            self.k_proj_option = owq()
            self.v_proj_option = owq()
            self.o_proj_option = build_qeft_w_options(bits['w'], self.n_qeft_column, eligible, with_outlier=False)
            self.gate_proj_option = owq()
            self.up_proj_option = owq()
            self.down_proj_option = owq()
        else:
            self.q_proj_option = bits['w']
            self.k_proj_option = bits['w']
            self.v_proj_option = bits['w']
            self.o_proj_option = bits['w']
            self.gate_proj_option = bits['w']
            self.up_proj_option = bits['w']
            self.down_proj_option = bits['w']

        # --- KV (bits, group_size) options ----------------------------------
        if 'k' in bits and 'k' in group_size:
            assert (len(group_size['k']) == 1 or len(group_size['k']) == len(bits['k']))
            if len(bits['k']) == 1:
                group_size['k'] = group_size['k'] * len(bits['k'])
            group_size['k'] = {b: sorted(g, reverse=True) for b, g in zip(bits['k'], group_size['k'])}
            self.k_option = [(b, g) for b, g_list in group_size['k'].items() for g in g_list]

        if 'v' in bits and 'v' in group_size:
            assert (len(group_size['v']) == 1 or len(group_size['v']) == len(bits['v']))
            if len(bits['v']) == 1:
                group_size['v'] = group_size['v'] * len(bits['v'])
            group_size['v'] = {b: sorted(g, reverse=True) for b, g in zip(bits['v'], group_size['v'])}
            self.v_option = [(b, g) for b, g_list in group_size['v'].items() for g in g_list]

        self.group_size = group_size
        self.pass_module = pass_module
        self.config = config
        self.n_linear = len(config['linear'])
        self.n_block = int(config['n_block'])

        self.comp_obj = comp_obj
        self.comp_obj_min = comp_obj_min
        self.comp_obj_max = comp_obj_max
        self.n_token = n_token

        # --- ThinK KV channel-pruning options -------------------------------
        head_dim = int(config['head_dim'])
        default_dims = sorted(set(round(r * head_dim) for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        self.k_pruning_dim_option = sorted(set(int(d) for d in (k_pruning_dim if k_pruning_dim is not None else default_dims)))
        self.v_pruning_dim_option = sorted(set(int(d) for d in (v_pruning_dim if v_pruning_dim is not None else default_dims)))

        # --- Frozen (single-option / pass_module) positions in the flat encoding
        # Layout rows: [w(n_linear) | k | v | k_dim | v_dim], each n_block wide.
        self.pass_idx_list = []
        for i, linear in enumerate(config['linear']):
            name = linear.split('.')[-1]
            if len(getattr(self, f'{name}_option')) == 1:
                self.pass_idx_list += list(range(i * self.n_block, (i + 1) * self.n_block))

        if len(self.k_option) == 1:
            self.pass_idx_list += list(range(self.n_linear * self.n_block, (self.n_linear + 1) * self.n_block))
        if len(self.v_option) == 1:
            self.pass_idx_list += list(range((self.n_linear + 1) * self.n_block, (self.n_linear + 2) * self.n_block))

        for pass_linear in self.pass_module['w']:
            blk, linear = pass_linear.split('.', maxsplit=1)
            linear_idx = self.config['linear'].index(linear)
            if len(getattr(self, f'{linear.split(".")[-1]}_option')) > 1:
                self.pass_idx_list.append(int(blk) + self.n_block * linear_idx)
        # K row is row n_linear, V row is row n_linear+1; block b sits at row*n_block + b.
        for pass_layer in self.pass_module['k']:
            if len(self.k_option) > 1:
                self.pass_idx_list.append(self.n_block * self.n_linear + pass_layer)
        for pass_layer in self.pass_module['v']:
            if len(self.v_option) > 1:
                self.pass_idx_list.append(self.n_block * (self.n_linear + 1) + pass_layer)

        if len(self.k_pruning_dim_option) == 1:
            self.pass_idx_list += list(range((self.n_linear + 2) * self.n_block, (self.n_linear + 3) * self.n_block))
        if len(self.v_pruning_dim_option) == 1:
            self.pass_idx_list += list(range((self.n_linear + 3) * self.n_block, (self.n_linear + 4) * self.n_block))

        self.pass_idx_list = sorted(set(self.pass_idx_list))

        # prob[] is indexed by option position during sampling, so it must be at
        # least as long as the largest option list (QEFT tuples can exceed rand_size).
        self.rand_size = max(rand_size,
                             len(self.q_proj_option), len(self.gate_proj_option),
                             len(self.o_proj_option), len(self.k_option), len(self.v_option))

        print(f'w_outlier : {self.w_outlier}, n_qeft_column : {self.n_qeft_column}')
        print(f'q_proj_option : {self.q_proj_option}')
        print(f'k_pruning_dim_option : {self.k_pruning_dim_option}')
        print(f'v_pruning_dim_option : {self.v_pruning_dim_option}')
        print(f'pass_idx_list : {self.pass_idx_list}')

    # ------------------------------------------------------------------ helpers
    def _w_opt_index(self, val, proj_name):
        """Index of a weight option value within the proj's option list,
        handling both scalar-bit and (bits, n_outlier)-tuple options."""
        opts = np.array(getattr(self, f'{proj_name}_option'))
        if self.w_outlier:
            return int(np.argwhere((np.array(val) == opts).all(axis=1))[0, 0])
        return int(np.argwhere(val == opts)[0, 0])

    def boundary_w_per_linear(self, w_option):
        """Map one boundary weight option (from the first linear's option list)
        to a per-linear forced-option list for sample(). With QEFT tuples, o_proj
        has no outlier variant, so (b, c>0) falls back to (b, 0)."""
        cols = []
        for linear in self.config['linear']:
            opts = getattr(self, f'{linear.split(".")[-1]}_option')
            if self.w_outlier:
                b, c = w_option
                cols.append([(b, c) if (b, c) in opts else (b, 0)])
            else:
                cols.append([w_option])
        return cols

    # ------------------------------------------------------------------ sampling
    def sample(self, n_samples=1, nb=None, w=None, k=None, v=None,
               k_dim=None, v_dim=None, pool=[]):
        """Randomly sample architectures in the canonical arch format."""
        nb = self.n_block if nb is None else nb

        w_q    = self.q_proj_option    if w is None else w[0]
        w_k    = self.k_proj_option    if w is None else w[1]
        w_v    = self.v_proj_option    if w is None else w[2]
        w_o    = self.o_proj_option    if w is None else w[3]
        w_gate = self.gate_proj_option if w is None else w[4]
        w_up   = self.up_proj_option   if w is None else w[5]
        w_down = self.down_proj_option if w is None else w[6]

        kv_k       = self.k_option             if k     is None else k
        kv_v       = self.v_option             if v     is None else v
        k_dim_opts = self.k_pruning_dim_option if k_dim is None else k_dim
        v_dim_opts = self.v_pruning_dim_option if v_dim is None else v_dim

        w_per_proj = {
            'q_proj': w_q, 'k_proj': w_k, 'v_proj': w_v, 'o_proj': w_o,
            'gate_proj': w_gate, 'up_proj': w_up, 'down_proj': w_down,
        }

        data = []
        for _ in tqdm(range(n_samples), desc='Sampling'):
            while True:
                prob = np.random.rand(self.rand_size)

                sampled = {}
                for proj_name, options in w_per_proj.items():
                    p = prob[np.array([self._w_opt_index(_x, proj_name) for _x in options])]
                    p = p / p.sum()
                    if self.w_outlier:
                        sel = np.random.choice(len(options), size=nb, p=p, replace=True)
                        sampled[proj_name] = [list(options[i]) for i in sel]
                    else:
                        sampled[proj_name] = np.random.choice(options, size=nb, p=p, replace=True).tolist()

                kv_k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in kv_k])]
                kv_k_list = np.array(kv_k)[np.random.choice(len(kv_k), size=nb, p=kv_k_prob / kv_k_prob.sum(), replace=True)].tolist()

                kv_v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in kv_v])]
                kv_v_list = np.array(kv_v)[np.random.choice(len(kv_v), size=nb, p=kv_v_prob / kv_v_prob.sum(), replace=True)].tolist()

                k_dim_list = np.random.choice(k_dim_opts, size=nb, replace=True).tolist()
                v_dim_list = np.random.choice(v_dim_opts, size=nb, replace=True).tolist()

                # Apply pass_module constraints (freeze at max option)
                for linear in self.pass_module['w']:
                    blk = int(linear.split('.')[0])
                    proj_name = linear.split('.')[-1]
                    mx = max(getattr(self, f'{proj_name}_option'))
                    sampled[proj_name][blk] = list(mx) if self.w_outlier else mx
                for layer in self.pass_module['k']:
                    kv_k_list[layer] = max(self.k_option, key=lambda x: (x[0], x[1]))
                for layer in self.pass_module['v']:
                    kv_v_list[layer] = max(self.v_option, key=lambda x: (x[0], x[1]))

                w_per_layer = {linear: sampled[linear.split('.')[-1]] for linear in self.config['linear']}

                new_arch = {
                    'q': {
                        'w': w_per_layer,
                        'k': [list(kv_k_list[i]) for i in range(nb)],
                        'v': [list(kv_v_list[i]) for i in range(nb)],
                    },
                    'p': {'k': k_dim_list, 'v': v_dim_list},
                }

                complexity = get_net_info(new_arch, self.config, self.group_size, n_token=self.n_token)
                flag = (new_arch not in data) and (new_arch not in pool)
                for i, obj in enumerate(self.comp_obj):
                    flag &= (math.isclose(complexity[obj], self.comp_obj_min[i]) or complexity[obj] > self.comp_obj_min[i]) and \
                            (math.isclose(complexity[obj], self.comp_obj_max[i]) or complexity[obj] < self.comp_obj_max[i])
                if flag:
                    break

            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        """Seed the archive with boundary configs (w × k × v × k_dim × v_dim),
        then fill the remaining budget with random samples."""
        data = []
        first_linear_option = getattr(self, f'{self.config["linear"][0].split(".")[-1]}_option')

        for w_option in first_linear_option:
            for k_option in self.k_option:
                for v_option in self.v_option:
                    for kp in self.k_pruning_dim_option:
                        for vp in self.v_pruning_dim_option:
                            data.append(self.sample(
                                w=self.boundary_w_per_linear(w_option),
                                k=[k_option], v=[v_option], k_dim=[kp], v_dim=[vp],
                            )[0])
                            n_doe -= 1

        data.extend(self.sample(n_samples=n_doe, pool=pool + data))
        return data

    # ------------------------------------------------------------------ encode / decode
    def encode(self, arch):
        """Canonical arch dict → flat integer array, length (n_linear + 4) * n_block.
        Layout: [w..., k..., v..., k_dim..., v_dim...]"""
        q_arch, p_arch = arch['q'], arch['p']

        w_encode = np.concatenate([
            np.array([self._w_opt_index(bits, linear.split('.')[-1]) for bits in q_arch['w'][linear]])
            for linear in self.config['linear']
        ])
        k_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.k_option)).all(axis=1))[0, 0] for _x in q_arch['k']
        ])
        v_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.v_option)).all(axis=1))[0, 0] for _x in q_arch['v']
        ])
        k_dim_encode = np.array([self.k_pruning_dim_option.index(int(d)) for d in p_arch['k']])
        v_dim_encode = np.array([self.v_pruning_dim_option.index(int(d)) for d in p_arch['v']])

        return np.concatenate((w_encode, k_encode, v_encode, k_dim_encode, v_dim_encode))

    def decode(self, x):
        """Flat integer array → canonical arch dict (with separate p.k / p.v)."""
        x_reshape = x.reshape(self.n_linear + 4, self.n_block)

        w_per_layer = {
            linear: np.array(getattr(self, f'{linear.split(".")[-1]}_option'))[x_reshape[i]].tolist()
            for i, linear in enumerate(self.config['linear'])
        }
        k_list     = [list(self.k_option[x_reshape[self.n_linear][i]]) for i in range(self.n_block)]
        v_list     = [list(self.v_option[x_reshape[self.n_linear + 1][i]]) for i in range(self.n_block)]
        k_dim_list = [self.k_pruning_dim_option[i] for i in x_reshape[self.n_linear + 2]]
        v_dim_list = [self.v_pruning_dim_option[i] for i in x_reshape[self.n_linear + 3]]

        return {
            'q': {'w': w_per_layer, 'k': k_list, 'v': v_list},
            'p': {'k': k_dim_list, 'v': v_dim_list},
        }

    def encode_predictor(self, arch):
        """Filtered encoding for the surrogate — single-option and pass_module
        entries are excluded (kept in sync with decode_encode_predictor)."""
        q_arch, p_arch = arch['q'], arch['p']

        w_encode = np.concatenate([
            np.array([
                self._w_opt_index(bits, linear.split('.')[-1])
                for layer_idx, bits in enumerate(q_arch['w'][linear])
                if f'{layer_idx}.{linear}' not in self.pass_module['w']
            ])
            # Skip single-option linears (e.g. o_proj when QEFT forces (b,0) and w
            # has one bit-width): they are frozen in pass_idx_list, so including them
            # here would desync encode_predictor from decode_encode_predictor.
            if ('wbits' in self.comp_obj or 'memory' in self.comp_obj)
               and len(getattr(self, f'{linear.split(".")[-1]}_option')) > 1 else []
            for linear in self.config['linear']
        ])

        k_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.k_option)).all(axis=1))[0, 0]
            for layer_idx, _x in enumerate(q_arch['k']) if layer_idx not in self.pass_module['k']
        ]) if 'kvbits' in self.comp_obj or 'kbits' in self.comp_obj or 'memory' in self.comp_obj \
            or 'eff_kvbits' in self.comp_obj or 'eff_kbits' in self.comp_obj else []

        v_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.v_option)).all(axis=1))[0, 0]
            for layer_idx, _x in enumerate(q_arch['v']) if layer_idx not in self.pass_module['v']
        ]) if 'kvbits' in self.comp_obj or 'vbits' in self.comp_obj or 'memory' in self.comp_obj \
            or 'eff_kvbits' in self.comp_obj or 'eff_vbits' in self.comp_obj else []

        k_dim_encode = np.array([
            self.k_pruning_dim_option.index(int(d)) for d in p_arch['k']
        ]) if len(self.k_pruning_dim_option) > 1 else np.array([], dtype=int)
        v_dim_encode = np.array([
            self.v_pruning_dim_option.index(int(d)) for d in p_arch['v']
        ]) if len(self.v_pruning_dim_option) > 1 else np.array([], dtype=int)

        return np.concatenate((w_encode, k_encode, v_encode, k_dim_encode, v_dim_encode))

    def decode_encode_predictor(self, x):  # x : (batch_size, dim)
        return np.delete(x, self.pass_idx_list, axis=-1)


# Backward-compatible aliases — every former Llama search-space class now resolves
# to the single unified LlamaSearchSpace above.
LlamaGroupSizeSearchSpace = LlamaSearchSpace
LlamaGroupSizeQEFTSearchSpace = LlamaSearchSpace
LlamaThinKSearchSpace = LlamaSearchSpace
