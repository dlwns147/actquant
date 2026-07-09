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

    # comp_obj → per-layer knob groups it controls, for stratified DOE sampling
    # (w=weight bits, k/v=KV bits, kd/vd=ThinK prune dims). Each comp_obj gets its own
    # independent complexity level; a knob group inherits the level of the comp_obj that
    # drives it so that axis is spread evenly and independent axes stay decoupled.
    _COMP_GROUPS = {
        'wbits': ('w',), 'memory': ('w', 'k', 'v', 'kd', 'vd'),
        'kvbits': ('k', 'v'), 'kbits': ('k',), 'vbits': ('v',),
        'kvdim': ('kd', 'vd'), 'kdim': ('kd',), 'vdim': ('vd',),
        'eff_kvbits': ('k', 'v', 'kd', 'vd'), 'eff_kbits': ('k', 'kd'), 'eff_vbits': ('v', 'vd'),
    }
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

        self.rand_size = rand_size

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
            # scalar bits `b` (from a plain/non-QEFT archive, e.g. the other axis's W) ≡ (b, 0):
            # 0 outlier columns. Lets a QEFT search space encode plain-W archs too.
            v = val if isinstance(val, (list, tuple, np.ndarray)) else (val, 0)
            return int(np.argwhere((np.array(v) == opts).all(axis=1))[0, 0])
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
    def _pick_level(self, cand, t, nb, ascending=True):
        """Per-layer pick from the complexity-ordered `cand`: option index ~
        Binomial(len-1, t), so the arch mean tracks t and t=0/1 hit the exact
        bottom/top option. ascending=False mirrors t for the prune axes
        (larger prune index = less complexity)."""
        idx = np.random.binomial(len(cand) - 1, t if ascending else 1.0 - t, size=nb)
        return [cand[i] for i in idx]

    def sample(self, n_samples=1, nb=None, w=None, k=None, v=None,
               k_dim=None, v_dim=None, pool=[]):
        """Randomly sample architectures in the canonical arch format.

        Each arch draws one complexity level t ~ U[0,1] per comp_obj and picks
        per-layer options around it (_pick_level), spreading every comp_obj
        over its full [min, max] range instead of clustering at the CLT mean."""
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

        # One level per comp_obj; knob groups inherit the level of the comp_obj they
        # drive (_COMP_GROUPS): same-comp_obj groups co-vary (spread that axis),
        # different comp_objs stay independent (no diagonal DOE), ungrouped knobs get
        # their own level. Mixed bit+prune comp_objs split the level anti-diagonally
        # (bits = base+d, prune = base-d, d uniform over the feasible range): the axis
        # value still tracks base — full spread, exact corners — while d sweeps the
        # bits↔prune combinations that realise it.
        _bits, _prune = ('w', 'k', 'v'), ('kd', 'vd')
        comp_lvl = {}
        for o in self.comp_obj:
            grps = self._COMP_GROUPS.get(o, ())
            base = np.random.rand(n_samples)
            mixed = any(g in _bits for g in grps) and any(g in _prune for g in grps)
            d = (np.random.rand(n_samples) * 2 - 1) * np.minimum(base, 1 - base) if mixed else 0.0
            comp_lvl[o] = {'bits': base + d, 'prune': base - d}
        group_levels = {}
        for g in ('w', 'k', 'v', 'kd', 'vd'):
            drv = next((o for o in self.comp_obj if g in self._COMP_GROUPS.get(o, ())), None)
            role = 'prune' if g in _prune else 'bits'
            group_levels[g] = comp_lvl[drv][role] if drv is not None else np.random.rand(n_samples)

        data = []
        for j in tqdm(range(n_samples), desc='Sampling'):
            attempt = 0
            while True:
                # after 50 rejected attempts, re-roll the levels so the comp_obj filter can't deadlock
                if attempt < 50:
                    t_w, t_k, t_v = group_levels['w'][j], group_levels['k'][j], group_levels['v'][j]
                    t_kd, t_vd = group_levels['kd'][j], group_levels['vd'][j]
                else:
                    t_w, t_k, t_v, t_kd, t_vd = np.random.rand(5)

                sampled = {}
                for proj_name, options in w_per_proj.items():
                    picks = self._pick_level(options, t_w, nb)
                    sampled[proj_name] = [list(x) for x in picks] if self.w_outlier else picks

                kv_k_list  = self._pick_level(kv_k, t_k, nb)
                kv_v_list  = self._pick_level(kv_v, t_v, nb)
                k_dim_list = self._pick_level(k_dim_opts, t_kd, nb, ascending=False)
                v_dim_list = self._pick_level(v_dim_opts, t_vd, nb, ascending=False)

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
                attempt += 1

            data.append(new_arch)
        return data

    def anchor_options(self, options, n_levels=0):
        """Thin an ordered option list to n_levels evenly spaced anchors
        (n_levels=3 → both ends + middle). Option lists are monotone in
        effective complexity (k/v: bits asc with gs desc within each bit;
        pruning dims asc), so the true extremes are always kept.
        n_levels=0 (or a list no longer than n_levels) keeps the full list."""
        if n_levels <= 0 or len(options) <= n_levels:
            return list(options)
        idx = np.unique(np.linspace(0, len(options) - 1, n_levels).round().astype(int))
        return [options[i] for i in idx]

    def paired_anchors(self, options_a, options_b, n_levels=0):
        """Pair two ordered option lists along their shared complexity diagonal
        so the two axes co-vary (K with V, k_prune with v_prune) instead of
        forming a cartesian product. Both lists are monotone in effective
        complexity, so position i is the same relative level in each. After
        anchoring the lengths may differ, so pair by proportional position
        (the true min/max corners stay aligned). Returns a list of (a, b) pairs
        of length max(len(a), len(b))."""
        a = self.anchor_options(options_a, n_levels)
        b = self.anchor_options(options_b, n_levels)
        n = max(len(a), len(b))
        if n <= 1:
            return [(a[0], b[0])]
        return [
            (a[round(i * (len(a) - 1) / (n - 1))], b[round(i * (len(b) - 1) / (n - 1))])
            for i in range(n)
        ]

    def initialize(self, n_doe, pool=[], anchor_levels=0):
        """Seed the archive with boundary configs, then fill the remaining
        budget with random samples.

        K/V and k_dim/v_dim are seeded as *paired* (diagonal) axes rather than
        a full cartesian product: each K option co-varies with the V option at
        the same complexity level (likewise k_prune with v_prune). This cuts the
        seed count from w × k × v × k_dim × v_dim down to
        w × max(k, v) × max(k_dim, v_dim) (e.g. eff_kvbits 9×9×5×5=2025 → 9×5=45;
        with anchor_levels=3 → 3×3=9), leaving more of n_doe for random fill.

        anchor_levels > 0 thins each axis to that many evenly spaced options
        (3 = min/mid/max) before pairing; 0 keeps every option."""
        data = []
        first_linear_option = getattr(self, f'{self.config["linear"][0].split(".")[-1]}_option')

        for w_option in self.anchor_options(first_linear_option, anchor_levels):
            for k_option, v_option in self.paired_anchors(self.k_option, self.v_option, anchor_levels):
                for kp, vp in self.paired_anchors(self.k_pruning_dim_option, self.v_pruning_dim_option, anchor_levels):
                    data.append(self.sample(
                        w=self.boundary_w_per_linear(w_option),
                        k=[k_option], v=[v_option], k_dim=[kp], v_dim=[vp],
                    )[0])
                    n_doe -= 1

        data.extend(self.sample(n_samples=max(n_doe, 0), pool=pool + data))
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


# ======================================================================
# QEFT outlier-column weight-search space (formerly search_space/llama_qeft.py)
# ======================================================================
# Adds a per-layer **QEFT outlier-column count** axis to the weight-quantization
# search. Every searchable linear's option becomes a ``(w_bits, n_outlier)``
# tuple, where ``n_outlier`` is the number of FP16 outlier *columns* kept for
# that layer (QEFT / OWQ). For hardware-friendly packing the count is restricted
# to multiples of 32 — by default ``[0, 32, 64, 96, 128]``.
#
# Unlike the unified LlamaSearchSpace above (which also exposes a QEFT axis via
# ``n_qeft_column`` alongside KV group-size + ThinK pruning), this class produces
# the legacy flat ``{'w':.., 'k':.., 'v':..}`` arch dict with a ``(n_linear + 2)``
# encoding (no ThinK ``'p'`` rows). It is kept for the QEFT-only weight search and
# its tests; the arch dict it produces is consumed unchanged by
# ``utils/func.py::get_net_info`` (``compute_bits`` / ``compute_memory`` already
# add ``out_dim * n_outlier * {16 bits | 2 bytes}`` for the FP16 outlier columns),
# so ``wbits`` / ``memory`` complexity objectives stay monotone with the new axis.
#
# Outlier columns are applied to qkv / gate / up / down (the OWQ layers in
# ``extract_outidx*.py``); ``o_proj`` is fixed to ``n_outlier = 0`` to match the
# extraction scripts (``meta['sequential'][1]`` is excluded there).

# QEFT keeps outlier columns in FP16; for kernel-friendly packing the count is a
# multiple of 32. This is the default; override via the ``n_qeft_column`` arg.
DEFAULT_QEFT_COLUMNS = [0, 32, 64, 96, 128]

# Linears that receive outlier columns (everything except o_proj), keyed by the
# trailing module name. Mirrors meta['sequential'][0]+[2]+[3] in extract_outidx*.
OUTLIER_LINEARS = {'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'}


def build_w_options(w_bits, n_qeft_column, outlier_bits=None, with_outlier=True):
    """Build a de-duplicated, order-preserving list of ``(bits, n_outlier)``.

    ``outlier_bits`` is the set of weight bit-widths eligible for outlier
    columns; if ``None`` it defaults to every sub-16-bit width (16-bit weights
    are already lossless, so spending FP16 outliers on them is wasteful and
    would only create ``(16, c)`` duplicates).  ``with_outlier=False`` forces the
    plain ``(bits, 0)`` axis (used for ``o_proj``).
    """
    w_bits = list(w_bits)
    if outlier_bits is None:
        outlier_bits = {b for b in w_bits if b < 16}
    else:
        outlier_bits = set(outlier_bits)

    options = []
    for b in w_bits:
        cols = n_qeft_column if (with_outlier and b in outlier_bits) else [0]
        for c in cols:
            if (b, c) not in options:
                options.append((b, c))
    return options


class LlamaQEFTSearchSpace:
    """Weight + KV search space with a per-layer QEFT outlier-column axis.

    Layout matches the other Llama search spaces: a flat encoded vector of shape
    ``(n_linear + 2) * n_block`` — one row per linear, then K, then V — so it is
    a drop-in for the per-axis search driver.  Each W entry is a ``(bits,
    n_outlier)`` pair; each K/V entry is a ``(bits, group_size)`` pair.
    """

    def __init__(self,
                 bits,
                 group_size,
                 pass_module,
                 config=None,
                 comp_obj='wbits',
                 comp_obj_min=[],
                 comp_obj_max=[],
                 outlier_bits=None,
                 n_qeft_column=DEFAULT_QEFT_COLUMNS,
                 n_token=0,
                 rand_size=5,
                 **kwargs):

        assert 'w' in pass_module and 'k' in pass_module and 'v' in pass_module

        # sanity: outlier counts must be non-negative multiples of 32.
        for c in n_qeft_column:
            assert c >= 0 and c % 32 == 0, f'n_qeft_column must be multiples of 32, got {c}'
        self.n_qeft_column = list(n_qeft_column)

        w_bits = bits['w']
        self.q_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits)
        self.k_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits)
        self.v_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits)
        self.o_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits, with_outlier=False)
        self.gate_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits)
        self.up_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits)
        self.down_proj_option = build_w_options(w_bits, self.n_qeft_column, outlier_bits)

        # KV (bits, group_size) options — same construction as LlamaSearchSpace.
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

        # Frozen positions in the encoded vector (single-option axes + pass_module).
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
        # K row is row n_linear, V row is row n_linear+1 in the (n_linear+2, n_block)
        # reshape, so block b sits at flat index row*n_block + b (no +1 offset).
        for pass_layer in self.pass_module['k']:
            if len(self.k_option) > 1:
                self.pass_idx_list.append(self.n_block * self.n_linear + pass_layer)
        for pass_layer in self.pass_module['v']:
            if len(self.v_option) > 1:
                self.pass_idx_list.append(self.n_block * (self.n_linear + 1) + pass_layer)

        self.pass_idx_list = sorted(set(self.pass_idx_list))
        self.rand_size = rand_size
        self.n_token = n_token

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _sample_axis(options, nb):
        """Sample ``nb`` entries from a list of tuple-options by index.

        Uses a per-call random weighting (non-uniform, like the other spaces)
        and indexes the option list rather than ``np.random.choice``-ing the
        tuples directly — choice() requires 1-D and chokes on a list of tuples.
        """
        w = np.random.rand(len(options))
        w = w / w.sum()
        idx = np.random.choice(len(options), size=nb, p=w, replace=True)
        # emit lists (not tuples): utils/func.py::compute_bits keys on
        # ``type(linear_bits[0]) == list`` for the (bits, n_outlier) W axis.
        return [list(options[i]) for i in idx]

    @staticmethod
    def _max_option(options):
        # max by (bits, n_outlier/group_size) — the "least compressed" corner.
        return list(max(options, key=lambda x: (x[0], x[1])))

    # ----------------------------------------------------------------- sample
    def sample(self, n_samples=1, nb=None, w=None, k=None, v=None, pool=[]):
        nb = self.n_block if nb is None else nb

        w_opts = {
            'q_proj': self.q_proj_option, 'k_proj': self.k_proj_option,
            'v_proj': self.v_proj_option, 'o_proj': self.o_proj_option,
            'gate_proj': self.gate_proj_option, 'up_proj': self.up_proj_option,
            'down_proj': self.down_proj_option,
        }
        # per-linear candidate pools (fixed value if w/k/v passed in)
        cand = {name: (w_opts[name] if w is None else w[i]) for i, name in enumerate(
            ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])}
        kv_k = self.k_option if k is None else k
        kv_v = self.v_option if v is None else v

        data = []
        for _ in tqdm(range(n_samples), desc='Sampling'):
            while True:
                w_lists = {name: self._sample_axis(cand[name], nb) for name in cand}
                kv_k_list = self._sample_axis(kv_k, nb)
                kv_v_list = self._sample_axis(kv_v, nb)

                # freeze pass_module weight layers at their max option
                for pass_linear in self.pass_module['w']:
                    blk, linear_name = pass_linear.split('.')[0], pass_linear.split('.')[-1]
                    w_lists[linear_name][int(blk)] = self._max_option(getattr(self, f'{linear_name}_option'))
                for layer in self.pass_module['k']:
                    kv_k_list[layer] = self._max_option(self.k_option)
                for layer in self.pass_module['v']:
                    kv_v_list[layer] = self._max_option(self.v_option)

                new_arch = {
                    'w': {
                        'self_attn.q_proj': w_lists['q_proj'], 'self_attn.k_proj': w_lists['k_proj'],
                        'self_attn.v_proj': w_lists['v_proj'], 'self_attn.o_proj': w_lists['o_proj'],
                        'mlp.gate_proj': w_lists['gate_proj'], 'mlp.up_proj': w_lists['up_proj'],
                        'mlp.down_proj': w_lists['down_proj'],
                    },
                    'k': kv_k_list,
                    'v': kv_v_list,
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
        data = []
        first_name = self.config['linear'][0].split('.')[-1]
        for w_option in getattr(self, f'{first_name}_option'):
            for k_option, v_option in zip(self.k_option, self.v_option):
                data.append(self.sample(w=[[w_option] for _ in self.config['linear']],
                                        k=[k_option], v=[v_option])[0])
                n_doe -= 1
        if n_doe > 0:
            data.extend(self.sample(n_samples=n_doe, pool=pool + data))
        return data

    # ----------------------------------------------------------------- encode
    @staticmethod
    def _index_of(value, options):
        opts = np.array(options)
        return int(np.argwhere((np.array(value) == opts).all(axis=1))[0, 0])

    def encode(self, arch):
        w_encode = np.concatenate([
            np.array([self._index_of(_x, getattr(self, f'{linear.split(".")[-1]}_option'))
                      for _x in arch['w'][linear]])
            for linear in self.config['linear']
        ])
        k_encode = np.array([self._index_of(_x, self.k_option) for _x in arch['k']])
        v_encode = np.array([self._index_of(_x, self.v_option) for _x in arch['v']])
        return np.concatenate((w_encode, k_encode, v_encode))

    def encode_predictor(self, arch):
        w_encode = np.concatenate([
            np.array([self._index_of(_x, getattr(self, f'{linear.split(".")[-1]}_option'))
                      for blk_idx, _x in enumerate(arch['w'][linear])
                      if f'{blk_idx}.{linear}' not in self.pass_module['w']])
            if 'wbits' in self.comp_obj or 'memory' in self.comp_obj else []
            for linear in self.config['linear']
        ])
        k_encode = np.array([self._index_of(_x, self.k_option)
                             for blk_idx, _x in enumerate(arch['k']) if blk_idx not in self.pass_module['k']]) \
            if 'kvbits' in self.comp_obj or 'kbits' in self.comp_obj or 'memory' in self.comp_obj else []
        v_encode = np.array([self._index_of(_x, self.v_option)
                             for blk_idx, _x in enumerate(arch['v']) if blk_idx not in self.pass_module['v']]) \
            if 'kvbits' in self.comp_obj or 'vbits' in self.comp_obj or 'memory' in self.comp_obj else []
        return np.concatenate((w_encode, k_encode, v_encode))

    def decode_encode_predictor(self, x):
        return np.delete(x, self.pass_idx_list, axis=-1)

    def decode(self, x):
        x_reshape = x.reshape(self.n_linear + 2, self.n_block)
        return {
            'w': {
                linear: np.array(getattr(self, f'{linear.split(".")[-1]}_option'))[x_reshape[i]].tolist()
                for i, linear in enumerate(self.config['linear'])
            },
            'k': np.array(self.k_option)[x_reshape[self.n_linear]].tolist(),
            'v': np.array(self.v_option)[x_reshape[self.n_linear + 1]].tolist(),
        }


# Backward-compatible aliases — every former Llama search-space class now resolves
# to the single unified LlamaSearchSpace above.
LlamaGroupSizeSearchSpace = LlamaSearchSpace
LlamaGroupSizeQEFTSearchSpace = LlamaSearchSpace
LlamaThinKSearchSpace = LlamaSearchSpace
