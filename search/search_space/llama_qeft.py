"""QEFT outlier-column weight-search space.

Adds a per-layer **QEFT outlier-column count** axis to the weight-quantization
search.  Every searchable linear's option becomes a ``(w_bits, n_outlier)``
tuple, where ``n_outlier`` is the number of FP16 outlier *columns* kept for that
layer (QEFT / OWQ).  For hardware-friendly packing the count is restricted to
multiples of 32 — by default ``[0, 32, 64, 96, 128]``.

Why a new file instead of editing ``LlamaGroupSizeQEFTSearchSpace`` (in
``search_space/llama.py``): that class was copied from the *scalar-W* parent and
its W-tuple handling is broken — ``np.random.choice`` on a list of tuples raises
("a must be 1-dimensional"), ``encode`` is missing the ``.all(axis=1)`` it uses
for the KV tuples, and the option builder emits duplicate ``(b, 0)`` entries for
non-outlier bit-widths.  This file fixes all three by treating the W axis exactly
like the already-correct KV ``(bits, group_size)`` axis: sample by *index*,
match by row, and de-duplicate the option list.

The arch dict this space produces is consumed unchanged by
``utils/func.py::get_net_info`` (``compute_bits`` / ``compute_memory`` already
add ``out_dim * n_outlier * {16 bits | 2 bytes}`` for the FP16 outlier columns),
so ``wbits`` / ``memory`` complexity objectives stay monotone with the new axis.

Outlier columns are applied to qkv / gate / up / down (the OWQ layers in
``extract_outidx*.py``); ``o_proj`` is fixed to ``n_outlier = 0`` to match the
extraction scripts (``meta['sequential'][1]`` is excluded there).
"""

import math

import numpy as np
from tqdm import tqdm

from utils.func import get_net_info


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

        # KV (bits, group_size) options — same construction as LlamaGroupSizeSearchSpace.
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
            for k_option in self.k_option:
                for v_option in self.v_option:
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
