import os
import numpy as np
import math
from utils.func import *
from utils.data import get_loader
from utils.eval import eval_metric, get_logits, get_tokenizer
from utils.loss import get_key_token_list
from model.replace import replace_kv_cache
from quant.model import get_quantized_model

# from model.skip_llama import block_replace
# from monkeypatch.ftllama_modeling import convert_model_to_ft
# from monkeypatch.ftllama_generate import replace_generate_functions

class LlamaEvaluator:
    def __init__(self,  
                 config,
                 accelerator,
                 method={},
                 model_id='',
                 quant_model_paths=[],
                #  quant_model_bits=[],
                 outlier=None,
                 datasets=['wikitext2'],
                 data_batch_size=1,
                 seed=0,
                 seqlen=2048,
                 min_seqlen=0,
                 n_sample=128,
                 n_token=0,
                 device_map='auto',
                 dtype='auto',
                #  dtype=torch.float16,
                 loss_func='cross_entropy',
                 inference=False,
                 bits={},
                 group_size={},
                 residual_length=128,
                 quant_kv_output=False,
                 k_quant_scheme='channel',
                 v_quant_scheme='token',
                 packing=False,
                #  use_flash=False,
                 limit=20,
                 num_fewshot=None,
                 lm_eval_batch_size=1,
                 task_manager=None,
                 task_dict=None,
                 verbosity='FATAL',
                 use_key_token=False,
                #  key_token_save_path='',
                 key_token_path='',
                 trunc_len=512,
                 sliding_window=128,
                 alpha=2,
                 beta=-2,
                 **kwargs):
        
        # model_id = os.path.join(model_path, model_name)
        self.method = method
        self.model = None
        self.group_size = group_size
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.tokenizer = get_tokenizer(model_id)
        n_block = int(config['n_block'])
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)

        # with accelerator.main_process_first():
        self.train_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, batch_size=data_batch_size, train=True, seed=seed, seqlen=seqlen, min_seqlen=min_seqlen)) for dataset in datasets}
        self.test_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, batch_size=data_batch_size, train=False, seed=seed, seqlen=seqlen, min_seqlen=min_seqlen)) for dataset in datasets}

        self.loss_func = loss_func
        self.dense_logits = {dataset: None for dataset in self.train_loaders.keys()}
        self.key_token_list = {dataset: None for dataset in self.train_loaders.keys()}
        self.outlier = dict()
        
        if loss_func in ['jsd', 'kld', 'topk'] or outlier is not None or use_key_token:
            # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map=device_map, low_cpu_mem_usage=True)
            model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)

            if loss_func in ['jsd', 'kld', 'topk']:
                self.dense_logits = {dataset: get_logits(model, loader) for dataset, loader in self.train_loaders.items()}

            if outlier is not None:
                for blk_idx in range(n_block):
                    for linear_group in config['linear']:
                        for linear in linear_group.split(','):
                            key = f'{config["layers"]}.{blk_idx}.{linear}'
                            if key in outlier:
                                self.outlier[f'{blk_idx}.{linear}'] = [outlier[key], get_fp16_channel(getsubattr(getblock(model, config)[blk_idx], linear), outlier[key])]

            if use_key_token:
                key_token_path_list = {dataset: os.path.join(key_token_path, dataset) for dataset in self.train_loaders}
                self.key_token_list = {
                    dataset: get_key_token_list(
                        evaluator_model=model,
                        evaluator_tokenizer=get_tokenizer(model_id, use_fast=True),
                        loader=loader,
                        trunc_len=trunc_len, 
                        sliding_window=sliding_window, 
                        alpha=alpha, 
                        beta=beta, 
                        load_path=key_token_path_list[dataset],
                        mode='offline'
                    ) for dataset, loader in self.train_loaders.items()
                }
                for dataset in self.train_loaders:
                    n_key_token = sum([len(key_token) for key_token in self.key_token_list[dataset]])
                    n_key_token = sum(accelerator.gather_for_metrics([n_key_token], use_gather_object=True))
                    accelerator.print(f'dataset: {dataset}, n_key_token: {n_key_token}')
                    accelerator.wait_for_everyone()

            del model
            clean_up()

        self.quant_models = list()
        if 'hqq' in method['w']:
            self.quant_model_bits = bits['w']
            if quant_model_paths and bits['w']:
                # with accelerator.main_process_first():
                self.model = load_hqq_model(quant_model_paths[np.argmax(bits['w'])], device_map, inference)

                if self.method['kv'] in ['hqq', 'kivi'] and (('k' in bits and 'v' in bits and max(bits['k']) < 16 and max(bits['v']) < 16)):
                    self.model = replace_kv_cache(model=self.model,
                                                tokenizer=self.tokenizer,
                                                method=method['kv'],
                                                n_block=n_block,
                                                k_quant_scheme=k_quant_scheme,
                                                v_quant_scheme=v_quant_scheme,
                                                residual_length=residual_length,
                                                packing=packing,
                                                quant_kv_output=quant_kv_output)

                self.remove_linears(self.model, config)
                self.quant_models = [load_hqq_model(p, device_map) for p in quant_model_paths]                
        
        elif 'fp16' in method['w']:
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)
            self.model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
            
            if self.method['kv'] in ['hqq', 'kivi'] and (('k' in bits and 'v' in bits and max(bits['k']) < 16 and max(bits['v']) < 16)):
                self.model = replace_kv_cache(model=self.model,
                                            tokenizer=self.tokenizer,
                                            method=method['kv'],
                                            n_block=n_block,
                                            k_quant_scheme=k_quant_scheme,
                                            v_quant_scheme=v_quant_scheme,
                                            residual_length=residual_length,
                                            packing=packing,
                                            quant_kv_output=quant_kv_output)
                
        elif not ('awq' in method['w'] or 'gptq' in method['w'] or 'qeft' in method['w']):
            raise NotImplementedError(method['w'])

        # if 'layer_prune' in method and self.model is not None:
        #     self.model = block_replace(self.model)
        #     self.model = simple_dispatch_model(self.model, device_map)

        self.config = config
        # self.latency_table = latency_table
        self.seqlen = seqlen
        
        if self.model is not None:
            self.model.eval()
            self.model.config.use_cache = False
            
        for q_model in self.quant_models:
            if q_model is not None:
                q_model.eval()
                q_model.config.use_cache = False

        self.n_token = n_token
        self.limit = limit
        self.num_fewshot = num_fewshot
        self.lm_eval_batch_size = lm_eval_batch_size
        self.task_manager = task_manager
        self.task_dict = task_dict
        self.verbosity = verbosity
        
        self.k_quant_scheme = k_quant_scheme
        self.v_quant_scheme = v_quant_scheme
        self.residual_length = residual_length
        self.quant_kv_output = quant_kv_output
        self.packing = packing
        
        self.use_key_token = use_key_token
        self.key_token_path = key_token_path
        self.trunc_len = trunc_len
        self.sliding_window = sliding_window
        self.alpha = alpha
        self.beta = beta

        accelerator.wait_for_everyone()

    def sample(self, arch):
        # self.validate_arch(arch)
        # if 'hqq' in self.method['w'] or 'awq' in self.method['w'] or 'gptq' in self.method['w'] or 'qeft' in self.method['w']:
        if 'hqq' in self.method['w']:
            for linear_group, linear_group_bits in arch['w'].items():
                for blk_idx, bits in enumerate(linear_group_bits):
                    flag = False
                    for q_bits, q_model in zip(self.quant_model_bits, self.quant_models):
                        # if math.isclose(bits, q_bits):
                        if math.isclose(int(bits), q_bits) and q_bits > 0:
                            for linear in linear_group.split(','):
                                # setsubattr(getblock(self.model, self.config)[blk_idx], linear, deepcopy(getsubattr(getblock(q_model, self.config)[blk_idx], linear)))
                                setsubattr(getblock(self.model, self.config)[blk_idx], linear, getsubattr(getblock(q_model, self.config)[blk_idx], linear))
                            flag = True

                    if not math.isclose(bits - int(bits), 0):
                        for linear in linear_group.split(','):
                            # insert_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear), self.outlier[f'{self.config["layers"]}.{blk_idx}.{linear}'])
                            insert_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear), self.outlier[f'{blk_idx}.{linear}'])
                    else:
                        for linear in linear_group.split(','):
                            remove_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear))

                    if not flag:
                        raise NotImplementedError(f'{linear_group}: {linear_group_bits} is not available')
                    
        elif 'awq' in self.method['w'] or 'gptq' in self.method['w'] or 'qeft' in self.method['w']:
            w_method = 'awq' if 'awq' in self.method['w'] else 'gptq' if 'gptq' in self.method['w'] else 'qeft' if 'qeft' in self.method['w'] else None
            self.model = get_quantized_model(method=w_method,
                                             arch=arch,
                                             model_name=self.model_id,
                                             device_map=self.device_map,
                                             group_size=self.group_size['w'],
                                             dtype=self.dtype,
                                             config=self.config,
                                             do_owq='qeft' in self.method['w'],
                                             owq_path=self.outlier)
            # self.model = get_hfmodel(self.model_id, self.device_map, self.dtype, use_cache=False)
            self.model.eval()
            # if (('k' in bits and 'v' in bits and max(bits['k']) < 16 and max(bits['v']) < 16)):
            if self.method['kv'] in ['hqq', 'kivi']:
                self.model = replace_kv_cache(model=self.model,
                                            tokenizer=self.tokenizer,
                                            method=self.method['kv'],
                                            n_block=int(self.config['n_block']),
                                            k_quant_scheme=self.k_quant_scheme,
                                            v_quant_scheme=self.v_quant_scheme,
                                            residual_length=self.residual_length,
                                            packing=self.packing,
                                            quant_kv_output=self.quant_kv_output)
        
        if self.method['kv'] == 'hqq':
            if 'k' in arch:
                self.model.generation_config.cache_config['k_bits'] = [x[0] for x in arch['k']]
                self.model.generation_config.cache_config['k_group_size'] = [x[1] for x in arch['k']]
            if 'v' in arch:
                self.model.generation_config.cache_config['v_bits'] = [x[0] for x in arch['v']]
                self.model.generation_config.cache_config['v_group_size'] = [x[1] for x in arch['v']]
        elif self.method['kv'] == 'kivi':
            if 'k' in arch:
                self.model.config.kivi_config.k_bits = [x[0] for x in arch['k']]
                self.model.config.kivi_config.k_group_size = [x[1] for x in arch['k']]

            if 'v' in arch:
                self.model.config.kivi_config.v_bits = [x[0] for x in arch['v']]
                self.model.config.kivi_config.v_group_size = [x[1] for x in arch['v']]
        elif self.method['kv'] == 'fp16':
            pass
        else:
            raise NotImplementedError(self.method['kv'])
        
        return self.model
    
    # def validate_arch(self, arch):
    #     assert all([l in self.config['linear'] for l in list(arch.keys())]), f'{list(arch.keys())} are invalid'
    #     for linear, linear_bits in arch.items():
    #         assert len(linear_bits) == self.config['n_block'], f'{linear}: len(linear_bits) != n_block'
    #         _, linear = linear.split('.')
    #         assert all([b in [0, self.small_model_bits, self.large_model_bits] for b in linear_bits]), f'{linear}: {linear_bits} are not compatible with the evaluator.'

    def eval(self, accelerator, arch, metric, model=None, loss_func='cross_entropy'):
        if metric == 'ppl':
            loaders = self.test_loaders
        elif metric == 'loss':
            loaders = self.train_loaders
        elif 'gsm8k' in metric:
            loaders = {metric: None}
        else:
            raise NotImplementedError(f"metric should be 'ppl', 'loss', or 'gsm8k', not {metric}")
        
        metric_list = dict()
        for dataset, loader in loaders.items():
            result = eval_metric(
                model=self.sample(arch) if model is None else model, 
                accelerator=accelerator,
                metric=metric, 
                loader=loader, 
                seqlen=self.seqlen, 
                loss_func=loss_func, 
                dense_logits_list=self.dense_logits[dataset] if (self.loss_func in ['jsd', 'kld', 'topk']) else None, 
                key_token_list=self.key_token_list[dataset] if self.use_key_token else None, 
                tokenizer=self.tokenizer,
                num_fewshot=self.num_fewshot, 
                limit=self.limit,
                batch_size=self.lm_eval_batch_size,
                verbosity=self.verbosity,
                task_manager=self.task_manager,
                task_dict=self.task_dict
            )
            
            if 'gsm8k' in metric:
                result = 1 - float(result[metric]['exact_match,strict-match'])
                # result = 1 - float(result[metric]['exact_match,flexible-extract'])
            
            metric_list[dataset] = result
        
        complexity = get_net_info(arch, self.config, self.group_size, n_token=self.n_token)
        return metric_list, complexity
    
    def remove_linears(self, model, config):
        for blk in getblock(model, config):
            for linear_group in config['linear']:
                for linear in linear_group.split(','):
                    delsubattr(blk, linear)
        clean_up()
