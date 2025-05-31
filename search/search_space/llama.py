import numpy as np
from utils.func import get_net_info
from tqdm import tqdm
import math


class LlamaSearchSpace:
    def __init__(self, 
                n_block,
                bits,
                group_size,
                pass_module,
                # min_bits,
                config=None,
                comp_obj='bits',
                comp_obj_min=[],
                comp_obj_max=[],
                outlier_bits=[],
                only_outlier_bits=False,
                rand_size=5
                ):
        self.n_block = n_block  # number of blocks
        
        # self.q_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.q_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.q_proj'])
        # self.k_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.k_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.k_proj'])
        # self.v_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.v_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.v_proj'])
        # # self.v_proj_option = sorted(quant_model_bits + outlier_bits['self_attn.v_proj'])
        # self.o_proj_option = quant_model_bits

        # self.gate_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.gate_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.gate_proj'])
        # self.up_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.up_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.up_proj'])
        # self.down_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.down_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.down_proj'])

        self.group_size = group_size
        # self.w_group_size = group_size['w']
        # # self.a_group_size = group_size['a']
        # self.v_group_size = group_size['v']
        # self.k_group_size = group_size['k']

        # self.w_bits = bits['w']
        # self.abits = bits['a']
        # self.k_bits = bits['k']
        # self.v_bits = bits['v']

        self.q_proj_option = bits['w']
        self.k_proj_option = bits['w']
        self.v_proj_option = bits['w']
        self.o_proj_option = bits['w']

        self.gate_proj_option = bits['w']
        self.up_proj_option = bits['w']
        self.down_proj_option = bits['w']

        # self.qkv_option = self.abits
        # self.o_option = self.abits
        # self.gateup_option = self.abits
        # self.down_option = self.abits

        self.k_option = bits['k']
        self.v_option = bits['v']
        
        self.pass_module = pass_module
        assert 'w' in pass_module and 'k' in pass_module and 'v' in pass_module
        self.config = config
        self.n_linear = len(config['linear'])
        self.n_layer = int(config['n_layer'])
        
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

        # self.n_bits = len(quant_model_bits)

    def sample(self, n_samples=1, nb=None, w=None, k=None, v=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb

        # assert len(act) == 4, f'len(act) : {len(act)}'
        w_q = self.q_proj_option if w is None else w[0]
        w_k = self.k_proj_option if w is None else w[1]
        w_v = self.v_proj_option if w is None else w[2]
        w_o = self.o_proj_option if w is None else w[3]
        w_gate = self.gate_proj_option if w is None else w[4]
        w_up = self.up_proj_option if w is None else w[5]
        w_down = self.down_proj_option if w is None else w[6]

        # a_qkv = self.qkv_option if act is None else act[0]
        # a_o = self.o_option if act is None else act[1]
        # a_gateup = self.gateup_option if act is None else act[2]
        # a_down = self.down_option if act is None else act[3]

        kv_k = self.k_option if k is None else k
        kv_v = self.v_option if v is None else v
        
        data = []
        # import pdb; pdb.set_trace()
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

                # a_qkv_prob = prob[np.array([np.argwhere(_x == np.array(self.qkv_option))[0, 0] for _x in a_qkv])]
                # a_qkv_list = np.random.choice(a_qkv, size=nb, p=a_qkv_prob / a_qkv_prob.sum(), replace=True).tolist()

                # a_o_prob = prob[np.array([np.argwhere(_x == np.array(self.o_option))[0, 0] for _x in a_o])]
                # a_o_list = np.random.choice(a_o, size=nb, p=a_o_prob / a_o_prob.sum(), replace=True).tolist()
                
                # a_gateup_prob = prob[np.array([np.argwhere(_x == np.array(self.gateup_option))[0, 0] for _x in a_gateup])]
                # a_gateup_list = np.random.choice(a_gateup, size=nb, p=a_gateup_prob / a_gateup_prob.sum(), replace=True).tolist()
                
                # a_down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_option))[0, 0] for _x in a_down])]
                # a_down_list = np.random.choice(a_down, size=nb, p=a_down_prob / a_down_prob.sum(), replace=True).tolist()
                
                kv_k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in kv_k])]
                kv_k_list = np.random.choice(kv_k, size=nb, p=kv_k_prob / kv_k_prob.sum(), replace=True).tolist()

                kv_v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in kv_v])]
                kv_v_list = np.random.choice(kv_v, size=nb, p=kv_v_prob / kv_v_prob.sum(), replace=True).tolist()

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
                
                # for linear in self.pass_module['activation']:
                #     blk, linear_name = linear.split('.')[0], linear.split('.')[-1]
                #     blk = int(blk)
                #     if linear_name == 'qkv':
                #         a_qkv_list[blk] = self.max_bits['a']
                #     elif linear_name == 'o':
                #         a_o_list[blk] = self.max_bits['a']
                #     elif linear_name == 'gateup':
                #         a_gateup_list[blk] = self.max_bits['a']
                #     elif linear_name == 'down':
                #         a_down_list[blk] = self.max_bits['a']
                #     else:
                        # raise NotImplementedError(f"linear : {linear}")

                for layer in self.pass_module['k']:
                    kv_k_list[layer] = max(self.k_option)

                for layer in self.pass_module['v']:
                    kv_v_list[layer] = max(self.v_option)
                    
                new_arch = {
                    'w': {'self_attn.q_proj': w_q_list, 'self_attn.k_proj': w_k_list, 'self_attn.v_proj': w_v_list, 'self_attn.o_proj': w_o_list, 'mlp.gate_proj': w_gate_list, 'mlp.up_proj': w_up_list, 'mlp.down_proj': w_down_list},
                    # 'activation' : {'qkv': a_qkv_list, 'o': a_o_list, 'gateup': a_gateup_list, 'down': a_down_list},
                    # 'kv_cache': {'k': kv_k_list, 'v': kv_v_list}
                    'k': kv_k_list,
                    'v': kv_v_list
                }
                complexity = get_net_info(new_arch, self.config, self.group_size)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
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
        data.append(self.sample(w=[[min(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[min(self.k_option)], v=[min(self.v_option)])[0])
        n_doe -= 1
        data.append(self.sample(w=[[max(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[max(self.k_option)], v=[max(self.v_option)])[0])
        n_doe -= 1
        if len(self.comp_obj) > 1:
            data.append(self.sample(w=[[min(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[max(self.k_option)], v=[max(self.v_option)])[0])
            n_doe -= 1
            data.append(self.sample(w=[[max(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[min(self.k_option)], v=[min(self.v_option)])[0])
            n_doe -= 1
        
        # for w_bits in self.q_proj_option:
        #     data.append(self.sample(w=[[w_bits]] * self.n_linear, k=[min(self.k_option)], v=[min(self.v_option)])[0])
        #     data.append(self.sample(w=[[w_bits]] * self.n_linear, k=[max(self.k_option)], v=[max(self.v_option)])[0])
        #     data.append(self.sample(w=[[w_bits]] * self.n_linear)[0])
        #     n_doe -= 3
        
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        w_encode = np.concatenate([
                np.array([
                    np.argwhere(_x == np.array(getattr(self, f'{linear.split(".")[-1]}_option')))[0, 0] for _x in arch['w'][linear] 
                    # np.argwhere(_x == np.array(self.w_bits))[0, 0] for _x in arch['w'][linear]
                    ]) for linear in self.config['linear']
            ])
        # a_encode = np.concatenate([np.array([np.argwhere(_x == np.array(self.abits))[0, 0] for _x in arch['activation'][linear_group]]) for linear_group in self.config['linear_group']])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in arch['k']])
        v_encode = np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in arch['v']])
        
        return np.concatenate((w_encode, k_encode, v_encode))
        # return np.concatenate((w_encode, a_encode, k_encode, v_encode))
    
    def encode_predictor(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        w_encode = np.concatenate([
                np.array([
                    np.argwhere(_x == np.array(getattr(self, f'{linear.split(".")[-1]}_option')))[0, 0] for blk_idx, _x in enumerate(arch['w'][linear]) if f'{blk_idx}.{linear}' not in self.pass_module['w'] 
                    ]) if 'wbits' in self.comp_obj else [] for linear in self.config['linear']
            ])
        # a_encode = np.concatenate([np.array([np.argwhere(_x == np.array(self.abits))[0, 0] for _x in arch['activation'][linear_group]]) for linear_group in self.config['linear_group']])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for blk_idx, _x in enumerate(arch['k']) if blk_idx not in self.pass_module['k']]) if 'kvbits' in self.comp_obj or 'kbits' in self.comp_obj else []
        v_encode = np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for blk_idx, _x in enumerate(arch['v']) if blk_idx not in self.pass_module['v']]) if 'kvbits' in self.comp_obj or 'vbits' in self.comp_obj else []

        # import pdb; pdb.set_trace()

        return np.concatenate((w_encode, k_encode, v_encode))
        # return np.delete(np.concatenate((w_encode, k_encode, v_encode)), self.pass_idx_list, axis=-1)

    def decode_encode_predictor(self, x): # x : (batch_size, dim)
        return np.delete(x, self.pass_idx_list, axis=-1)

    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_linear + 2, self.n_block)
        return {
                    'w': {
                        linear: np.array(getattr(self, f'{linear.split(".")[-1]}_option'))[x_reshape[i]].tolist() for i, linear in enumerate(self.config["linear"])
                    },
                    'k': np.array(self.k_option)[x_reshape[self.n_linear]].tolist(),
                    'v': np.array(self.v_option)[x_reshape[self.n_linear + 1]].tolist(),
                }
    


class LlamaGroupsizeSearchSpace:
    def __init__(self, 
                n_block,
                bits,
                group_size,
                pass_module,
                # min_bits,
                config=None,
                comp_obj='bits',
                comp_obj_min=[],
                comp_obj_max=[],
                outlier_bits=[],
                only_outlier_bits=False,
                rand_size=5
                ):
        self.n_block = n_block  # number of blocks
        
        # self.q_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.q_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.q_proj'])
        # self.k_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.k_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.k_proj'])
        # self.v_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.v_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.v_proj'])
        # # self.v_proj_option = sorted(quant_model_bits + outlier_bits['self_attn.v_proj'])
        # self.o_proj_option = quant_model_bits

        # self.gate_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.gate_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.gate_proj'])
        # self.up_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.up_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.up_proj'])
        # self.down_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.down_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.down_proj'])

        self.group_size = group_size
        # self.w_group_size = group_size['w']
        # # self.a_group_size = group_size['a']
        # self.v_group_size = group_size['v']
        # self.k_group_size = group_size['k']

        # self.w_bits = bits['w']
        # self.abits = bits['a']
        # self.k_bits = bits['k']
        # self.v_bits = bits['v']

        self.q_proj_option = bits['w']
        self.k_proj_option = bits['w']
        self.v_proj_option = bits['w']
        self.o_proj_option = bits['w']

        self.gate_proj_option = bits['w']
        self.up_proj_option = bits['w']
        self.down_proj_option = bits['w']

        # self.qkv_option = self.abits
        # self.o_option = self.abits
        # self.gateup_option = self.abits
        # self.down_option = self.abits

        self.k_option = bits['k']
        self.v_option = bits['v']
        
        self.pass_module = pass_module
        assert 'w' in pass_module and 'k' in pass_module and 'v' in pass_module
        self.config = config
        self.n_linear = len(config['linear'])
        self.n_layer = int(config['n_layer'])
        
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


    def sample(self, n_samples=1, nb=None, w=None, k=None, v=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb

        # assert len(act) == 4, f'len(act) : {len(act)}'
        w_q = self.q_proj_option if w is None else w[0]
        w_k = self.k_proj_option if w is None else w[1]
        w_v = self.v_proj_option if w is None else w[2]
        w_o = self.o_proj_option if w is None else w[3]
        w_gate = self.gate_proj_option if w is None else w[4]
        w_up = self.up_proj_option if w is None else w[5]
        w_down = self.down_proj_option if w is None else w[6]

        kv_k = self.k_option if k is None else k
        kv_v = self.v_option if v is None else v
        
        data = []
        # import pdb; pdb.set_trace()
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
                kv_k_list = np.random.choice(kv_k, size=nb, p=kv_k_prob / kv_k_prob.sum(), replace=True).tolist()

                kv_v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in kv_v])]
                kv_v_list = np.random.choice(kv_v, size=nb, p=kv_v_prob / kv_v_prob.sum(), replace=True).tolist()

                # kv_k_gs_prob = prob[np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in kv_k])]
                # kv_k_gs_list = np.random.choice(kv_k, size=nb, p=kv_k_prob / kv_k_prob.sum(), replace=True).tolist()

                # kv_v_gs_prob = prob[np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in kv_v])]
                # kv_v_gs_list = np.random.choice(kv_v, size=nb, p=kv_v_prob / kv_v_prob.sum(), replace=True).tolist()

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
                    kv_k_list[layer] = max(self.k_option)

                for layer in self.pass_module['v']:
                    kv_v_list[layer] = max(self.v_option)
                    
                new_arch = {
                    'w': {'self_attn.q_proj': w_q_list, 'self_attn.k_proj': w_k_list, 'self_attn.v_proj': w_v_list, 'self_attn.o_proj': w_o_list, 'mlp.gate_proj': w_gate_list, 'mlp.up_proj': w_up_list, 'mlp.down_proj': w_down_list},
                    'k': kv_k_list,
                    'v': kv_v_list,
                    'group_size': {
                        'k': [],
                        'v': []
                    }
                }
                complexity = get_net_info(new_arch, self.config, self.group_size)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
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
        data.append(self.sample(w=[[min(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[min(self.k_option)], v=[min(self.v_option)])[0])
        n_doe -= 1
        data.append(self.sample(w=[[max(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[max(self.k_option)], v=[max(self.v_option)])[0])
        n_doe -= 1
        if len(self.comp_obj) > 1:
            data.append(self.sample(w=[[min(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[max(self.k_option)], v=[max(self.v_option)])[0])
            n_doe -= 1
            data.append(self.sample(w=[[max(getattr(self, f'{l.split(".")[-1]}_option'))] for l in self.config['linear']], k=[min(self.k_option)], v=[min(self.v_option)])[0])
            n_doe -= 1
        
        # for w_bits in self.q_proj_option:
        #     data.append(self.sample(w=[[w_bits]] * self.n_linear, k=[min(self.k_option)], v=[min(self.v_option)])[0])
        #     data.append(self.sample(w=[[w_bits]] * self.n_linear, k=[max(self.k_option)], v=[max(self.v_option)])[0])
        #     data.append(self.sample(w=[[w_bits]] * self.n_linear)[0])
        #     n_doe -= 3
        
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        w_encode = np.concatenate([
                np.array([
                    np.argwhere(_x == np.array(getattr(self, f'{linear.split(".")[-1]}_option')))[0, 0] for _x in arch['w'][linear] 
                    # np.argwhere(_x == np.array(self.w_bits))[0, 0] for _x in arch['w'][linear]
                    ]) for linear in self.config['linear']
            ])
        # a_encode = np.concatenate([np.array([np.argwhere(_x == np.array(self.abits))[0, 0] for _x in arch['activation'][linear_group]]) for linear_group in self.config['linear_group']])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for _x in arch['k']])
        v_encode = np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for _x in arch['v']])
        
        return np.concatenate((w_encode, k_encode, v_encode))
        # return np.concatenate((w_encode, a_encode, k_encode, v_encode))
    
    def encode_predictor(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        w_encode = np.concatenate([
                np.array([
                    np.argwhere(_x == np.array(getattr(self, f'{linear.split(".")[-1]}_option')))[0, 0] for blk_idx, _x in enumerate(arch['w'][linear]) if f'{blk_idx}.{linear}' not in self.pass_module['w'] 
                    ]) if 'wbits' in self.comp_obj else [] for linear in self.config['linear']
            ])
        # a_encode = np.concatenate([np.array([np.argwhere(_x == np.array(self.abits))[0, 0] for _x in arch['activation'][linear_group]]) for linear_group in self.config['linear_group']])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_option))[0, 0] for blk_idx, _x in enumerate(arch['k']) if blk_idx not in self.pass_module['k']]) if 'kvbits' in self.comp_obj or 'kbits' in self.comp_obj else []
        v_encode = np.array([np.argwhere(_x == np.array(self.v_option))[0, 0] for blk_idx, _x in enumerate(arch['v']) if blk_idx not in self.pass_module['v']]) if 'kvbits' in self.comp_obj or 'vbits' in self.comp_obj else []

        # import pdb; pdb.set_trace()

        return np.concatenate((w_encode, k_encode, v_encode))
        # return np.delete(np.concatenate((w_encode, k_encode, v_encode)), self.pass_idx_list, axis=-1)

    def decode_encode_predictor(self, x): # x : (batch_size, dim)
        return np.delete(x, self.pass_idx_list, axis=-1)

    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_linear + 2, self.n_block)
        return {
                    'w': {
                        linear: np.array(getattr(self, f'{linear.split(".")[-1]}_option'))[x_reshape[i]].tolist() for i, linear in enumerate(self.config["linear"])
                    },
                    'k': np.array(self.k_option)[x_reshape[self.n_linear]].tolist(),
                    'v': np.array(self.v_option)[x_reshape[self.n_linear + 1]].tolist(),
                }
    

