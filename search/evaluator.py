import numpy as np
import math
from utils.func import *
from utils.data import get_loader
from utils.eval import eval_metric, get_logits, eval_zeroshot, get_tokenizer
from model.replace import replace_model
 

# from model.skip_llama import block_replace
# from monkeypatch.ftllama_modeling import convert_model_to_ft
# from monkeypatch.ftllama_generate import replace_generate_functions

class LlamaEvaluator:
    def __init__(self,  
                 config,
                 accelerator,
                 method=[],
                 model_id='',
                 quant_model_paths=[],
                #  quant_model_bits=[],
                 outlier=None,
                 datasets=['wikitext2'],
                 data_batch_size=1,
                 seed=0,
                 seqlen=2048,
                 n_sample=128,
                 device_map='auto',
                #  dtype='auto',
                 dtype=torch.float16,
                 cache_dir=None,
                 loss_func='cross_entropy',
                 inference=False,
                 bits={},
                 group_size={},
                 residual_length=128,
                 quant_kv_output=False,
                 k_quant_per='channel',
                 v_quant_per='token',
                 use_flash=False,
                 limit=20,
                 num_fewshot=None,
                 lm_eval_batch_size=1,
                 task_manager=None,
                 task_dict=None,
                 verbosity='FATAL',
                 **kwargs):
        
        # model_id = os.path.join(model_path, model_name)
        self.method = method
        self.model = None
        self.group_size = group_size
        self.tokenizer = get_tokenizer(model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)

        # with accelerator.main_process_first():
        self.train_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, batch_size=data_batch_size, train=True, seed=seed, seqlen=seqlen)) for dataset in datasets}
        self.test_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, batch_size=data_batch_size, train=False, seqlen=seqlen)) for dataset in datasets}

        self.loss_func = loss_func
        self.outlier = dict()
        if loss_func == 'jsd' or outlier is not None:
            # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map=device_map, low_cpu_mem_usage=True)
            model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)

            if loss_func == 'jsd':
                self.dense_logits = {dataset: get_logits(model, loader) for dataset, loader in self.train_loaders.items()}

            if outlier is not None:
                for blk_idx in range(int(config['n_block'])):
                    for linear_group in config['linear']:
                        for linear in linear_group.split(','):
                            key = f'{config["layers"]}.{blk_idx}.{linear}'
                            if key in outlier:
                                self.outlier[f'{blk_idx}.{linear}'] = [outlier[key], get_fp16_channel(getsubattr(getblock(model, config)[blk_idx], linear), outlier[key])]
            del model
            clean_up()

        if loss_func != 'jsd':
            self.dense_logits = {dataset: None for dataset in self.train_loaders.keys()}

        self.quant_models = list()
        if 'hqq' in method:
            self.quant_model_bits = bits['w']
            if quant_model_paths and bits['w']:
                # with accelerator.main_process_first():
                self.model = load_hqq_model(quant_model_paths[np.argmax(bits['w'])], device_map, inference)

                if ('k' in bits or 'v' in bits):
                    self.model.config.k_bits = [max(bits['k'])] * config['n_block']
                    self.model.config.v_bits = [max(bits['v'])] * config['n_block']
                    self.model.config.k_group_size = [max(group_size['k'][-1])] * config['n_block']
                    self.model.config.v_group_size = [max(group_size['v'][-1])] * config['n_block']
                    # if len(group_size['k']) == 1:
                    #     self.model.config.k_group_size = [max(group_size['k'][-1])] * config['n_block']
                    # elif len(group_size['k']) >= 2:
                    #     self.model.config.k_group_size = [max(group_size['k'][-1])] * config['n_block']
                        
                    # if len(group_size['v']) == 1:
                    #     self.model.config.v_group_size = [max(group_size['v'])] * config['n_block']
                    # elif len(group_size['v']) >= 2:
                    #     self.model.config.v_group_size = [max(group_size['v'][-1])] * config['n_block']
                    self.model.config.use_flash = use_flash

                    self.model.config.residual_length = residual_length 
                    self.model.config.quant_kv_output = quant_kv_output
                    self.model.config.k_quant_per = k_quant_per
                    self.model.config.v_quant_per = v_quant_per

                    self.model = replace_model(self.model, self.model.config)

                self.remove_linears(self.model, config)
                self.quant_models = [load_hqq_model(p, device_map) for p in quant_model_paths]
        
        elif 'awq' in method or 'gptq' in method or 'owq' in method:
            pass

        else:
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)
            self.model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
            
            if ('k' in bits or 'v' in bits):
                self.model.config.k_bits = [max(bits['k'])] * config['n_block']
                self.model.config.v_bits = [max(bits['v'])] * config['n_block']
                
                self.model.config.k_group_size = [max(group_size['k'][-1])] * config['n_block']
                self.model.config.v_group_size = [max(group_size['v'][-1])] * config['n_block']
                # if len(group_size['k']) == 1:
                #     self.model.config.k_group_size = [max(group_size['k'])] * config['n_block']
                # elif len(group_size['k']) == 2:
                #     self.model.config.k_group_size = [max(group_size['k'][-1])] * config['n_block']
                    
                # if len(group_size['v']) == 1:
                #     self.model.config.v_group_size = [max(group_size['v'])] * config['n_block']
                # elif len(group_size['v']) == 2:
                #     self.model.config.v_group_size = [max(group_size['v'][-1])] * config['n_block']
                self.model.config.use_flash = use_flash

                self.model.config.residual_length = residual_length 
                self.model.config.quant_kv_output = quant_kv_output
                self.model.config.k_quant_per = k_quant_per
                self.model.config.v_quant_per = v_quant_per

                self.model = replace_model(self.model, self.model.config)

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

        self.limit = limit
        self.num_fewshot = num_fewshot
        self.lm_eval_batch_size = lm_eval_batch_size
        self.task_manager = task_manager
        self.task_dict = task_dict
        self.verbosity = verbosity
        accelerator.wait_for_everyone()

    def sample(self, arch):
        # self.validate_arch(arch)
        if 'hqq' in self.method or 'awq' in self.method or 'gptq' in self.method or 'owq' in self.method:
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
                    
        if 'k' in arch:
            if type(arch['k'][0]) in [int, float]:
                self.model.config.k_bits = arch['k']
            elif type(arch['k'][0]) is list:
                self.model.config.k_bits = [x[0] for x in arch['k']]
                self.model.config.k_group_size = [x[1] for x in arch['k']]

        if 'v' in arch:
            if type(arch['v'][0]) in [int, float]:
                self.model.config.v_bits = arch['v']
            elif type(arch['v'][0]) is list:
                self.model.config.v_bits = [x[0] for x in arch['v']]
                self.model.config.v_group_size = [x[1] for x in arch['v']]
            else:
                raise NotImplementedError
        
        return self.model
    
    # def validate_arch(self, arch):
    #     assert all([l in self.config['linear'] for l in list(arch.keys())]), f'{list(arch.keys())} are invalid'
    #     for linear, linear_bits in arch.items():
    #         assert len(linear_bits) == self.config['n_block'], f'{linear}: len(linear_bits) != n_block'
    #         _, linear = linear.split('.')
    #         assert all([b in [0, self.small_model_bits, self.large_model_bits] for b in linear_bits]), f'{linear}: {linear_bits} are not compatible with the evaluator.'

    def eval(self, accelerator, arch, metric, model=None, loss_func='cross_entropy'):
        # if metric == 'latency':
        #     measure_latency(model=self.sample(arch))
        if metric == 'ppl':
            loaders = self.test_loaders
        elif metric == 'loss':
            loaders = self.train_loaders
        elif 'gsm8k' in metric:
            loaders = {metric: None}
        else:
            raise NotImplementedError(f"metric should be 'ppl' or 'loss', not {metric}")
        metric_list = dict()
        for dataset, loader in loaders.items():
            result = eval_metric(model=self.sample(arch) if model is None else model, 
                                accelerator=accelerator,
                                metric=metric, 
                                loader=loader, 
                                seqlen=self.seqlen, 
                                loss_func=loss_func, 
                                dense_logits_list=self.dense_logits[dataset] if self.loss_func=='jsd' else None, 
                                num_fewshot=self.num_fewshot, 
                                limit=self.limit,
                                batch_size=self.lm_eval_batch_size,
                                verbosity=self.verbosity,
                                task_manager=self.task_manager,
                                task_dict=self.task_dict)
            if 'gsm8k' in metric:
                result = 1 - float(result[metric]['exact_match,strict-match'])
                # result = 1 - float(result[metric]['exact_match,flexible-extract'])
            metric_list[dataset] = result
        complexity = get_net_info(arch, self.config, self.group_size)
        return metric_list, complexity
    
    def remove_linears(self, model, config):
        for blk in getblock(model, config):
            for linear_group in config['linear']:
                for linear in linear_group.split(','):
                    delsubattr(blk, linear)
        clean_up()

    def eval_woo(self, accelerator, arch, model, metric, loss_func='cross_entropy'):
        # if metric == 'latency':
        #     measure_latency(model=self.sample(arch))
        if metric == 'ppl':
            loaders = self.test_loaders
        elif metric == 'loss':
            loaders = self.train_loaders
        else:
            raise NotImplementedError(f"metric should be 'ppl' or 'loss', not {metric}")
        metric_list = dict()
        for dataset, loader in loaders.items():
            metric_list[dataset] = eval_metric(model=model, accelerator=accelerator, metric=metric, loader=loader, seqlen=self.seqlen, loss_func=loss_func, dense_logits_list=self.dense_logits[dataset])
        complexity = get_net_info(arch, self.config, self.latency_table)
        # torch.cuda.empty_cache()
        return metric_list, complexity
