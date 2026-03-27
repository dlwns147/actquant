import numpy as np
from .llama import LlamaGroupSizeSearchSpace
from utils.func import get_net_info
from tqdm import tqdm
import math


class LlamaThinKSearchSpace(LlamaGroupSizeSearchSpace):
    """
    Search space for mixed-precision KV quantization with per-layer KV pruning.

    Uses the canonical arch schema (same as awqgptq.py):
        arch = {
            'q': {
                'w': {linear: [bits, ...], ...},   # per-linear dict of per-block bit lists
                'k': [[bits, group_size], ...],    # list of [bits, gs] per layer
                'v': [[bits, group_size], ...],
            },
            'p': {
                'k': [pruning_dim, ...],           # per-layer K pruning dim (int, # dims removed from head_dim)
                'v': [pruning_dim, ...],           # per-layer V pruning dim (int, # dims removed from head_dim)
            }
        }

    pruning_dim convention: 0 = no pruning, head_dim//2 = prune 50% of KV channels.
    Used by utils/func.py as: effective_bits = (bits + scale_overhead) * (1 - pruning_dim / head_dim)

    K and V pruning grids are configured independently via k_pruning_dim / v_pruning_dim
    (explicit integer lists of per-layer dims to prune).

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
                 ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            pass_module=pass_module,
            config=config,
            comp_obj=comp_obj,
            comp_obj_min=comp_obj_min,
            comp_obj_max=comp_obj_max,
            outlier_bits=outlier_bits,
            only_outlier_bits=only_outlier_bits,
            n_token=n_token,
            rand_size=rand_size,
        )

        head_dim = int(config['head_dim'])
        # Default: 6 evenly-spaced integer counts from 0 to head_dim//2
        default_dims = sorted(set(round(r * head_dim) for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        self.k_pruning_dim_option = sorted(set(int(d) for d in (k_pruning_dim if k_pruning_dim is not None else default_dims)))
        self.v_pruning_dim_option = sorted(set(int(d) for d in (v_pruning_dim if v_pruning_dim is not None else default_dims)))

        # Encoding indices:
        #   k_dim: [(n_linear+2)*n_block, (n_linear+3)*n_block)
        #   v_dim: [(n_linear+3)*n_block, (n_linear+4)*n_block)
        if len(self.k_pruning_dim_option) == 1:
            self.pass_idx_list += list(range(
                (self.n_linear + 2) * self.n_block,
                (self.n_linear + 3) * self.n_block,
            ))
        if len(self.v_pruning_dim_option) == 1:
            self.pass_idx_list += list(range(
                (self.n_linear + 3) * self.n_block,
                (self.n_linear + 4) * self.n_block,
            ))
        self.pass_idx_list.sort()

        print(f'k_pruning_dim_option : {self.k_pruning_dim_option}')
        print(f'v_pruning_dim_option : {self.v_pruning_dim_option}')
        print(f'pass_idx_list (after pruning) : {self.pass_idx_list}')

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

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

        kv_k         = self.k_option             if k     is None else k
        kv_v         = self.v_option             if v     is None else v
        k_dim_opts   = self.k_pruning_dim_option if k_dim is None else k_dim
        v_dim_opts   = self.v_pruning_dim_option if v_dim is None else v_dim

        w_per_proj = {
            'q_proj':    w_q,
            'k_proj':    w_k,
            'v_proj':    w_v,
            'o_proj':    w_o,
            'gate_proj': w_gate,
            'up_proj':   w_up,
            'down_proj': w_down,
        }

        data = []
        for _ in tqdm(range(n_samples), desc='Sampling'):
            while True:
                prob = np.random.rand(self.rand_size)

                sampled = {}
                for proj_name, options in w_per_proj.items():
                    p = prob[np.array([
                        np.argwhere(_x == np.array(getattr(self, f'{proj_name}_option')))[0, 0]
                        for _x in options
                    ])]
                    sampled[proj_name] = np.random.choice(
                        options, size=nb, p=p / p.sum(), replace=True
                    ).tolist()

                kv_k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in kv_k])]
                kv_k_list = np.array(kv_k)[np.random.choice(len(kv_k), size=nb, p=kv_k_prob / kv_k_prob.sum(), replace=True)].tolist()

                kv_v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in kv_v])]
                kv_v_list = np.array(kv_v)[np.random.choice(len(kv_v), size=nb, p=kv_v_prob / kv_v_prob.sum(), replace=True)].tolist()

                # Independent uniform choice for K and V pruning ratios
                k_dim_list = np.random.choice(k_dim_opts, size=nb, replace=True).tolist()
                v_dim_list = np.random.choice(v_dim_opts, size=nb, replace=True).tolist()

                # Apply pass_module constraints
                for linear in self.pass_module['w']:
                    blk = int(linear.split('.')[0])
                    proj_name = linear.split('.')[-1]
                    sampled[proj_name][blk] = max(getattr(self, f'{proj_name}_option'))

                for layer in self.pass_module['k']:
                    kv_k_list[layer] = max(self.k_option, key=lambda x: (x[0], x[1]))
                for layer in self.pass_module['v']:
                    kv_v_list[layer] = max(self.v_option, key=lambda x: (x[0], x[1]))

                w_per_layer = {
                    linear: sampled[linear.split('.')[-1]]
                    for linear in self.config['linear']
                }

                new_arch = {
                    'q': {
                        'w': w_per_layer,
                        'k': [list(kv_k_list[i]) for i in range(nb)],
                        'v': [list(kv_v_list[i]) for i in range(nb)],
                    },
                    'p': {
                        'k': k_dim_list,
                        'v': v_dim_list,
                    },
                }

                complexity = get_net_info(new_arch, self.config, self.group_size, n_token=self.n_token)
                flag = (new_arch not in data) and (new_arch not in pool)
                for i, obj in enumerate(self.comp_obj):
                    flag &= (
                        math.isclose(complexity[obj], self.comp_obj_min[i]) or
                        complexity[obj] > self.comp_obj_min[i]
                    ) and (
                        math.isclose(complexity[obj], self.comp_obj_max[i]) or
                        complexity[obj] < self.comp_obj_max[i]
                    )
                if flag:
                    break

            data.append(new_arch)
        return data

    # ------------------------------------------------------------------
    # Initialization (DOE boundary configs)
    # ------------------------------------------------------------------

    def initialize(self, n_doe, pool=[]):
        """
        Seed the archive with boundary configurations.
        Iterates (w × k × v × k_dim_options × v_dim_options).
        Remaining budget is filled with random samples.
        """
        data = []
        first_linear_option = getattr(self, f'{self.config["linear"][0].split(".")[-1]}_option')

        for w_option in first_linear_option:
            for k_option in self.k_option:
                for v_option in self.v_option:
                    for kp in self.k_pruning_dim_option:
                        for vp in self.v_pruning_dim_option:
                            data.append(self.sample(
                                w=[[w_option] for _ in self.config['linear']],
                                k=[k_option],
                                v=[v_option],
                                k_dim=[kp],
                                v_dim=[vp],
                            )[0])
                            n_doe -= 1

        data.extend(self.sample(n_samples=n_doe, pool=pool + data))
        return data

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, arch):
        """
        Encode canonical arch dict → flat integer array of length (n_linear + 4) * n_block.
        Layout: [w..., k..., v..., k_dim..., v_dim...]
        """
        q_arch, p_arch = arch['q'], arch['p']

        w_encode = np.concatenate([
            np.array([
                np.argwhere(
                    bits == np.array(getattr(self, f'{linear.split(".")[-1]}_option'))
                )[0, 0]
                for bits in q_arch['w'][linear]
            ])
            for linear in self.config['linear']
        ])
        k_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.k_option)).all(axis=1))[0, 0]
            for _x in q_arch['k']
        ])
        v_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.v_option)).all(axis=1))[0, 0]
            for _x in q_arch['v']
        ])
        k_dim_encode = np.array([
            self.k_pruning_dim_option.index(int(d)) for d in p_arch['k']
        ])
        v_dim_encode = np.array([
            self.v_pruning_dim_option.index(int(d)) for d in p_arch['v']
        ])

        return np.concatenate((w_encode, k_encode, v_encode, k_dim_encode, v_dim_encode))

    def decode(self, x):
        """
        Decode flat integer array → canonical arch dict with separate p.k / p.v.
        """
        x_reshape = x.reshape(self.n_linear + 4, self.n_block)

        w_per_layer = {
            linear: np.array(getattr(self, f'{linear.split(".")[-1]}_option'))[x_reshape[i]].tolist()
            for i, linear in enumerate(self.config['linear'])
        }
        k_list      = [list(self.k_option[x_reshape[self.n_linear    ][i]]) for i in range(self.n_block)]
        v_list      = [list(self.v_option[x_reshape[self.n_linear + 1][i]]) for i in range(self.n_block)]
        k_dim_list  = [self.k_pruning_dim_option[i] for i in x_reshape[self.n_linear + 2]]
        v_dim_list  = [self.v_pruning_dim_option[i] for i in x_reshape[self.n_linear + 3]]

        return {
            'q': {
                'w': w_per_layer,
                'k': k_list,
                'v': v_list,
            },
            'p': {
                'k': k_dim_list,
                'v': v_dim_list,
            },
        }

    def encode_predictor(self, arch):
        """
        Filtered encoding for surrogate model training.
        Single-option and pass_module entries are excluded.
        """
        q_arch, p_arch = arch['q'], arch['p']

        w_encode = np.concatenate([
            np.array([
                np.argwhere(
                    bits == np.array(getattr(self, f'{linear.split(".")[-1]}_option'))
                )[0, 0]
                for layer_idx, bits in enumerate(q_arch['w'][linear])
                if f'{layer_idx}.{linear}' not in self.pass_module['w']
            ]) if 'wbits' in self.comp_obj or 'memory' in self.comp_obj else []
            for linear in self.config['linear']
        ])

        k_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.k_option)).all(axis=1))[0, 0]
            for layer_idx, _x in enumerate(q_arch['k'])
            if layer_idx not in self.pass_module['k']
        ]) if 'kvbits' in self.comp_obj or 'kbits' in self.comp_obj or 'memory' in self.comp_obj \
            or 'eff_kvbits' in self.comp_obj or 'eff_kbits' in self.comp_obj else []

        v_encode = np.array([
            np.argwhere((np.array(_x) == np.array(self.v_option)).all(axis=1))[0, 0]
            for layer_idx, _x in enumerate(q_arch['v'])
            if layer_idx not in self.pass_module['v']
        ]) if 'kvbits' in self.comp_obj or 'vbits' in self.comp_obj or 'memory' in self.comp_obj \
            or 'eff_kvbits' in self.comp_obj or 'eff_vbits' in self.comp_obj else []

        k_dim_encode = np.array([
            self.k_pruning_dim_option.index(int(d)) for d in p_arch['k']
        ]) if len(self.k_pruning_dim_option) > 1 else np.array([], dtype=int)

        v_dim_encode = np.array([
            self.v_pruning_dim_option.index(int(d)) for d in p_arch['v']
        ]) if len(self.v_pruning_dim_option) > 1 else np.array([], dtype=int)

        return np.concatenate((w_encode, k_encode, v_encode, k_dim_encode, v_dim_encode))

    # decode_encode_predictor is inherited from parent:
    #   np.delete(x, self.pass_idx_list, axis=-1)
