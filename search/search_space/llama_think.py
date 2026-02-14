import numpy as np
from utils.func import get_net_info
from tqdm import tqdm
import math


class LlamaGroupSizeSearchSpaceThink:
    def __init__(self, 
                bits,
                group_size,
                pass_module,
                config=None,
                comp_obj='bits',
                comp_obj_min=[],
                comp_obj_max=[],
                outlier_bits=[],
                only_outlier_bits=False,
                n_token=0,
                rand_size=5,
                pruning_ratio_options=[0.25, 0.5, 0.75, 1.0]
                ):
        
        assert 'w' in pass_module and 'k' in pass_module and 'v' in pass_module

        self.q_proj_option = bits['w']
        self.k_proj_option = bits['w']
        self.v_proj_option = bits['w']
        self.o_proj_option = bits['w']

        self.gate_proj_option = bits['w']
        self.up_proj_option = bits['w']
        self.down_proj_option = bits['w']

        # KV cache quantization options
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

        # Pruning ratio options (retention rate)
        self.pruning_ratio_option = sorted(pruning_ratio_options)

        self.group_size = group_size
        self.pass_module = pass_module
        
        self.config = config
        self.n_linear = len(config['linear'])
        self.n_block = int(config['n_block'])
        
        self.comp_obj = comp_obj
        self.comp_obj_min = comp_obj_min
        self.comp_obj_max = comp_obj_max
        
        self.pass_idx_list = []
        for i, linear in enumerate(config['linear']):
            _, linear = linear.split('.')
            if len(getattr(self, f'{linear}_option')) == 1:
                self.pass_idx_list += list(range(i * self.n_block, (i + 1) * self.n_block))

        if len(self.k_option) == 1:
            self.pass_idx_list += list(range(self.n_linear * self.n_block, (self.n_linear + 1) * self.n_block))

        if len(self.v_option) == 1:
            self.pass_idx_list += list(range((self.n_linear + 1) * self.n_block, (self.n_linear + 2) * self.n_block))

        # Pruning ratio is per-layer, but we use the same value for all layers for simplicity
        # If len(pruning_ratio_option) == 1, we can add it to pass_idx_list
        # Note: pruning_ratio is encoded as a single value per architecture, not per layer

        for pass_linear in self.pass_module['w']:
            blk, linear = pass_linear.split('.', maxsplit=1)
            linear_idx = self.config['linear'].index(linear)
            if len(getattr(self, f'{linear.split(".")[-1]}_option')) > 1:
                self.pass_idx_list.append(int(blk) + self.n_block * linear_idx)

        for pass_layer in self.pass_module['k']:
            if len(self.k_option) > 1:
                self.pass_idx_list.append(self.n_block * self.n_linear + pass_layer + 1)

        for pass_layer in self.pass_module['v']:
            if len(self.v_option) > 1:
                self.pass_idx_list.append(self.n_block * (self.n_linear + 1) + pass_layer + 1)

        self.pass_idx_list.sort()
        print(f'self.pass_idx_list : {self.pass_idx_list}')
        self.rand_size = rand_size
        self.n_token = n_token


    def sample(self, n_samples=1, nb=None, w=None, k=None, v=None, pruning_ratio=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb

        w_q = self.q_proj_option if w is None else w[0]
        w_k = self.k_proj_option if w is None else w[1]
        w_v = self.v_proj_option if w is None else w[2]
        w_o = self.o_proj_option if w is None else w[3]
        w_gate = self.gate_proj_option if w is None else w[4]
        w_up = self.up_proj_option if w is None else w[5]
        w_down = self.down_proj_option if w is None else w[6]

        kv_k = self.k_option if k is None else k
        kv_v = self.v_option if v is None else v
        pruning_ratio_list = self.pruning_ratio_option if pruning_ratio is None else pruning_ratio
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:
                prob = np.random.rand(self.rand_size)

                w_q_prob = prob[np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in w_q])]
                w_q_list = np.random.choice(w_q, size=nb, p=w_q_prob / w_q_prob.sum(), replace=True).tolist()

                w_k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in w_k])]
                w_k_list = np.random.choice(w_k, size=nb, p=w_k_prob / w_k_prob.sum(), replace=True).tolist()
                
                w_v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in w_v])]
                w_v_list = np.random.choice(w_v, size=nb, p=w_v_prob / w_v_prob.sum(), replace=True).tolist()
                
                w_o_prob = prob[np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in w_o])]
                w_o_list = np.random.choice(w_o, size=nb, p=w_o_prob / w_o_prob.sum(), replace=True).tolist()

                w_gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in w_gate])]
                w_gate_list = np.random.choice(w_gate, size=nb, p=w_gate_prob / w_gate_prob.sum(), replace=True).tolist()

                w_up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in w_up])]
                w_up_list = np.random.choice(w_up, size=nb, p=w_up_prob / w_up_prob.sum(), replace=True).tolist()

                w_down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in w_down])]
                w_down_list = np.random.choice(w_down, size=nb, p=w_down_prob / w_down_prob.sum(), replace=True).tolist()

                kv_k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in kv_k])]
                kv_k_list = np.array(kv_k)[np.random.choice(len(kv_k), size=nb, p=kv_k_prob / kv_k_prob.sum(), replace=True)].tolist()

                kv_v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in kv_v])]
                kv_v_list = np.array(kv_v)[np.random.choice(len(kv_v), size=nb, p=kv_v_prob / kv_v_prob.sum(), replace=True)].tolist()

                # Sample pruning ratio (same for all layers for simplicity)
                pruning_ratio_val = np.random.choice(self.pruning_ratio_option) if pruning_ratio is None else pruning_ratio_list[0]
                pruning_ratio_list_sampled = [pruning_ratio_val] * nb

                for linear in self.pass_module['w']:
                    blk, linear_name = linear.split('.')[0], linear.split('.')[-1]
                    blk = int(blk)
                    if linear_name == 'q_proj':
                        w_q_list[blk] = max(self.q_proj_option)
                    elif linear_name == 'k_proj':
                        w_k_list[blk] = max(self.k_proj_option)
                    elif linear_name == 'v_proj':
                        w_v_list[blk] = max(self.v_proj_option)
                    elif linear_name == 'o_proj':
                        w_o_list[blk] = max(self.o_proj_option)
                    elif linear_name == 'gate_proj':
                        w_gate_list[blk] = max(self.gate_proj_option)
                    elif linear_name == 'up_proj':
                        w_up_list[blk] = max(self.up_proj_option)
                    elif linear_name == 'down_proj':
                        w_down_list[blk] = max(self.down_proj_option)
                    else:
                        raise NotImplementedError(f"linear : {linear}")
                
                for layer in self.pass_module['k']:
                    kv_k_list[layer] = max(self.k_option, key=lambda x: (x[0], x[1]))

                for layer in self.pass_module['v']:
                    kv_v_list[layer] = max(self.v_option, key=lambda x: (x[0], x[1]))
                    
                new_arch = {
                    'w': {'self_attn.q_proj': w_q_list, 'self_attn.k_proj': w_k_list, 'self_attn.v_proj': w_v_list, 'self_attn.o_proj': w_o_list, 'mlp.gate_proj': w_gate_list, 'mlp.up_proj': w_up_list, 'mlp.down_proj': w_down_list},
                    'k': kv_k_list,
                    'v': kv_v_list,
                    'pruning_ratio': pruning_ratio_list_sampled,  # Add pruning ratio to arch
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
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        for w_option in getattr(self, f'{self.config["linear"][0].split(".")[-1]}_option'):
            for k_option in self.k_option:
                for v_option in self.v_option:
                    for pruning_ratio_option in self.pruning_ratio_option:
                        data.append(self.sample(w=[[w_option] for _ in self.config['linear']], k=[k_option], v=[v_option], pruning_ratio=[pruning_ratio_option])[0])
                        n_doe -= 1
        
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch to integer bit-string
        w_encode = np.concatenate([
                np.array([
                    np.argwhere(_x == np.array(getattr(self, f'{linear.split(".")[-1]}_option')))[0, 0] for _x in arch['w'][linear] 
                    ]) for linear in self.config['linear']
            ])
        k_encode = np.array([np.argwhere((_x == np.array(self.k_option)).all(axis=1))[0, 0] for _x in arch['k']])
        v_encode = np.array([np.argwhere((_x == np.array(self.v_option)).all(axis=1))[0, 0] for _x in arch['v']])
        # Encode pruning ratio (same value for all layers, so just take the first one)
        pruning_ratio_val = arch['pruning_ratio'][0] if isinstance(arch['pruning_ratio'], list) else arch['pruning_ratio']
        pruning_ratio_encode = np.array([np.argwhere(pruning_ratio_val == np.array(self.pruning_ratio_option))[0, 0]] * self.n_block)
        
        return np.concatenate((w_encode, k_encode, v_encode, pruning_ratio_encode))
    
    def encode_predictor(self, arch):
        # encode arch for predictor (excluding pass modules)
        w_encode = np.concatenate([
                np.array([
                    np.argwhere(_x == np.array(getattr(self, f'{linear.split(".")[-1]}_option')))[0, 0] for blk_idx, _x in enumerate(arch['w'][linear]) if f'{blk_idx}.{linear}' not in self.pass_module['w'] 
                    ]) if 'wbits' in self.comp_obj or 'memory' in self.comp_obj else [] for linear in self.config['linear']
            ])
        k_encode = np.array([np.argwhere((_x == np.array(self.k_option)).all(axis=1))[0, 0] for blk_idx, _x in enumerate(arch['k']) if blk_idx not in self.pass_module['k']]) if 'kvbits' in self.comp_obj or 'kbits' in self.comp_obj or 'memory' in self.comp_obj else []
        v_encode = np.array([np.argwhere((_x == np.array(self.v_option)).all(axis=1))[0, 0] for blk_idx, _x in enumerate(arch['v']) if blk_idx not in self.pass_module['v']]) if 'kvbits' in self.comp_obj or 'vbits' in self.comp_obj or 'memory' in self.comp_obj else []
        # Encode pruning ratio (same value for all layers)
        pruning_ratio_val = arch['pruning_ratio'][0] if isinstance(arch['pruning_ratio'], list) else arch['pruning_ratio']
        pruning_ratio_encode = np.array([np.argwhere(pruning_ratio_val == np.array(self.pruning_ratio_option))[0, 0]])

        return np.concatenate((w_encode, k_encode, v_encode, pruning_ratio_encode))

    def decode_encode_predictor(self, x): # x : (batch_size, dim)
        return np.delete(x, self.pass_idx_list, axis=-1)

    def decode(self, x):
        # decode integer bit-string to arch
        x_reshape = x.reshape(self.n_linear + 3, self.n_block)  # +3 for k, v, pruning_ratio
        pruning_ratio_val = np.array(self.pruning_ratio_option)[x_reshape[self.n_linear + 2, 0]]  # Use first value for all layers
        return {
                    'w': {
                        linear: np.array(getattr(self, f'{linear.split(".")[-1]}_option'))[x_reshape[i]].tolist() for i, linear in enumerate(self.config["linear"])
                    },
                    'k': np.array(self.k_option)[x_reshape[self.n_linear]].tolist(),
                    'v': np.array(self.v_option)[x_reshape[self.n_linear + 1]].tolist(),
                    'pruning_ratio': [pruning_ratio_val] * self.n_block,  # Same pruning ratio for all layers
                }
