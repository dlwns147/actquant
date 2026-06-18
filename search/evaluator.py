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
                 attn_sink=0,
                 quant_kv_output=False,
                 k_quant_scheme='channel',
                 v_quant_scheme='token',
                 packing=False,
                #  use_flash=False,
                 k_pruning_dim=0,
                 v_pruning_dim=0,
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
                 last_tokens=None,
                 precomputed_train_loaders=None,
                 precomputed_test_loaders=None,
                 precomputed_dense_logits=None,
                 precomputed_key_token_list=None,
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
        # Loaders / dense_logits / key tokens can be injected (precomputed
        # upstream in ONE FP-teacher pass and reused across metric groups — see
        # correlation.py). Injecting the SAME loader objects the dense_logits
        # were computed over guarantees index alignment in eval_loss.
        if precomputed_train_loaders is not None:
            self.train_loaders = precomputed_train_loaders
            self.test_loaders = (precomputed_test_loaders
                                 if precomputed_test_loaders is not None else {})
        else:
            self.train_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, batch_size=data_batch_size, train=True, seed=seed, seqlen=seqlen, min_seqlen=min_seqlen)) for dataset in datasets}
            self.test_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, batch_size=data_batch_size, train=False, seed=seed, seqlen=seqlen, min_seqlen=min_seqlen)) for dataset in datasets}

        self.loss_func = loss_func
        self.dense_logits = (precomputed_dense_logits if precomputed_dense_logits is not None
                             else {dataset: None for dataset in self.train_loaders.keys()})
        self.key_token_list = (precomputed_key_token_list if precomputed_key_token_list is not None
                               else {dataset: None for dataset in self.train_loaders.keys()})
        self.outlier = dict()
        # Raw outlier dict ({model.layers.i.linear: [col indices]}) as loaded from
        # disk. self.outlier (below) is the HQQ-shaped {blk.linear: [idx, fp16ch]}
        # variant; the QEFT path (quant/qeft.py) needs the RAW form instead.
        self.outlier_raw = outlier
        self.last_tokens = last_tokens

        # Only spin up the FP teacher for work that is NOT already injected:
        # outlier fp16-channels (always needs it), key tokens, dense_logits.
        need_keytok = use_key_token and precomputed_key_token_list is None
        need_dense = (loss_func in ['jsd', 'kld', 'topk', 'forward_kl']) and precomputed_dense_logits is None
        if need_dense or need_keytok or outlier is not None:
            # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map=device_map, low_cpu_mem_usage=True)
            model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)

            # Only the HQQ path consumes the {blk.linear: [idx, fp16ch]} form and
            # needs flat per-layer index lists. QEFT uses the RAW dict (self.outlier_raw)
            # and may carry the multi-rank {n_out: [idx]} form, which would break the
            # get_fp16_channel indexing below — so skip this transform for non-HQQ.
            if outlier is not None and 'hqq' in method['w']:
                # Build the per-layer FP16 outlier payload(s) for the HQQ path. Each
                # payload is [col_indices, fp16_cols] (consumed by hqq dequant:
                # W_r[:, idx] = fp16_cols). Two source formats:
                #   * multi-rank {key: {n_out: [idx]}}  → {n_out: [idx, fp16_cols]}
                #     (searchable QEFT outlier-column axis; w arch = (bits, n_outlier))
                #   * flat {key: [idx]}                 → [idx, fp16_cols] (legacy)
                for blk_idx in range(n_block):
                    for linear_group in config['linear']:
                        for linear in linear_group.split(','):
                            key = f'{config["layers"]}.{blk_idx}.{linear}'
                            if key not in outlier:
                                continue
                            tlinear = getsubattr(getblock(model, config)[blk_idx], linear)
                            entry = outlier[key]
                            if isinstance(entry, dict):
                                self.outlier[f'{blk_idx}.{linear}'] = {
                                    int(n): [idx, get_fp16_channel(tlinear, idx)]
                                    for n, idx in entry.items()}
                            else:
                                self.outlier[f'{blk_idx}.{linear}'] = [entry, get_fp16_channel(tlinear, entry)]

            if need_keytok:
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

            if need_dense:
                self.dense_logits = {
                    dataset: get_logits(
                        model, loader,
                        key_token_list=self.key_token_list[dataset] if use_key_token else None,
                        last_tokens=self.last_tokens
                    ) for dataset, loader in self.train_loaders.items()
                }

            del model
            clean_up()

        self.quant_models = list()
        if 'hqq' in method['w']:
            self.quant_model_bits = bits['w']
            if quant_model_paths and bits['w']:
                # If the caller passed an explicit torch.dtype, force HQQ to
                # honour it (override its hard-coded fp16 default + the path's
                # config.json torch_dtype). For dtype='auto', load_hqq_model
                # falls back to the path's torch_dtype.
                _compute_dtype = dtype if isinstance(dtype, torch.dtype) else None
                self.model = load_hqq_model(quant_model_paths[np.argmax(bits['w'])],
                                            device_map, inference,
                                            compute_dtype=_compute_dtype)

                kv_methods = self.method['kv'] if isinstance(self.method.get('kv'), list) else [self.method.get('kv')]
                if any(m in ['hqq', 'kivi', 'think'] for m in kv_methods) and (('k' in bits and 'v' in bits and max(bits['k']) < 16 and max(bits['v']) < 16)):
                    self.model = replace_kv_cache(model=self.model,
                                                tokenizer=self.tokenizer,
                                                method=kv_methods,
                                                n_block=n_block,
                                                k_quant_scheme=k_quant_scheme,
                                                v_quant_scheme=v_quant_scheme,
                                                residual_length=residual_length,
                                                sink=attn_sink,
                                                packing=packing,
                                                quant_kv_output=quant_kv_output,
                                                k_pruning_dim=k_pruning_dim,
                                                v_pruning_dim=v_pruning_dim)

                self.remove_linears(self.model, config)
                self.quant_models = [load_hqq_model(p, device_map,
                                                    compute_dtype=_compute_dtype)
                                     for p in quant_model_paths]
        
        elif 'fp16' in method['w']:
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)
            self.model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
            
            kv_methods = self.method['kv'] if isinstance(self.method.get('kv'), list) else [self.method.get('kv')]
            if any(m in ['hqq', 'kivi', 'think'] for m in kv_methods) and (('k' in bits and 'v' in bits and max(bits['k']) < 16 and max(bits['v']) < 16)):
                self.model = replace_kv_cache(model=self.model,
                                            tokenizer=self.tokenizer,
                                            method=kv_methods,
                                            n_block=n_block,
                                            k_quant_scheme=k_quant_scheme,
                                            v_quant_scheme=v_quant_scheme,
                                            residual_length=residual_length,
                                            sink=attn_sink,
                                            packing=packing,
                                            quant_kv_output=quant_kv_output,
                                            k_pruning_dim=k_pruning_dim,
                                            v_pruning_dim=v_pruning_dim)
                
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
            # Resolve dtype='auto' → the model's actual torch.dtype so KIVI /
            # downstream consumers can rely on self.dtype being a real
            # torch.dtype rather than a string sentinel.
            if not isinstance(self.dtype, torch.dtype):
                try:
                    self.dtype = next(self.model.parameters()).dtype
                except StopIteration:
                    pass
            
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
        self.attn_sink = attn_sink
        self.quant_kv_output = quant_kv_output
        self.packing = packing
        self.k_pruning_dim = k_pruning_dim
        self.v_pruning_dim = v_pruning_dim
        
        self.use_key_token = use_key_token
        self.key_token_path = key_token_path
        self.trunc_len = trunc_len
        self.sliding_window = sliding_window
        self.alpha = alpha
        self.beta = beta

        accelerator.wait_for_everyone()

    def sample(self, arch):
        # arch schema: {'q': {'w': {linear: [bits,...],...}, 'k': [[bits,gs],...], 'v': [...]}, 'p': {'k': [dim,...], 'v': [...]}}
        q_arch = arch['q']
        p_arch = arch.get('p', {})

        if 'hqq' in self.method['w']:
            n_block = len(next(iter(q_arch['w'].values())))
            for blk_idx in range(n_block):
                for linear_group in self.config['linear']:
                    entry = q_arch['w'][linear_group][blk_idx]
                    # (bits, n_outlier) tuple (QEFT outlier-column axis) or scalar bits.
                    # Outlier trigger: n_outlier>0 (tuple) or fractional bits (legacy).
                    if isinstance(entry, (list, tuple)):
                        bits, n_out = int(entry[0]), int(entry[1])
                        want_outlier = n_out > 0
                    else:
                        bits, n_out = int(entry), 0
                        want_outlier = not math.isclose(entry - int(entry), 0)

                    flag = False
                    for q_bits, q_model in zip(self.quant_model_bits, self.quant_models):
                        if math.isclose(bits, q_bits) and q_bits > 0:
                            for linear in linear_group.split(','):
                                setsubattr(getblock(self.model, self.config)[blk_idx], linear, getsubattr(getblock(q_model, self.config)[blk_idx], linear))
                            flag = True
                    if not flag:
                        raise NotImplementedError(f'{linear_group}: {entry} is not available')

                    # Insert FP16 outlier columns on top of the HQQ bank (or clear).
                    for linear in linear_group.split(','):
                        ln = getsubattr(getblock(self.model, self.config)[blk_idx], linear)
                        if want_outlier:
                            od = self.outlier[f'{blk_idx}.{linear}']
                            insert_fp16_channel_hqq(ln, od[n_out] if isinstance(od, dict) else od)
                        else:
                            remove_fp16_channel_hqq(ln)

        elif 'awq' in self.method['w'] or 'gptq' in self.method['w'] or 'qeft' in self.method['w']:
            w_method = 'awq' if 'awq' in self.method['w'] else 'gptq' if 'gptq' in self.method['w'] else 'qeft'
            # AWQ / GPTQ / QEFT load via BASE.load_model →
            # from_pretrained(torch_dtype=self.dtype). dtype='auto' picks up
            # the model config's torch_dtype (e.g. Llama-3.1-8B → bf16);
            # passing torch.float16 / torch.bfloat16 forces that exact dtype.
            # Weight pseudo-quant in awq_utils/quantizer.py is dtype-agnostic
            # so the activation/KV dtype is preserved end-to-end.
            self.model = get_quantized_model(method=w_method,
                                             arch=q_arch['w'],
                                             model_name=self.model_id,
                                             device_map=self.device_map,
                                             group_size=self.group_size['w'],
                                             dtype=self.dtype,
                                             config=self.config,
                                             do_owq='qeft' in self.method['w'],
                                             owq_path=self.outlier_raw)
            self.model.eval()
            # Resolve self.dtype → real torch.dtype now that the model is up;
            # downstream KIVI cache + cuda_bmm dispatch reads model.dtype but
            # other call sites (e.g. configure_model_cache) compare self.dtype.
            if not isinstance(self.dtype, torch.dtype):
                try:
                    self.dtype = next(self.model.parameters()).dtype
                except StopIteration:
                    pass
            kv_methods = self.method['kv'] if isinstance(self.method.get('kv'), list) else [self.method.get('kv')]
            if any(m in ['hqq', 'kivi', 'think'] for m in kv_methods):
                self.model = replace_kv_cache(model=self.model,
                                            tokenizer=self.tokenizer,
                                            method=kv_methods,
                                            n_block=int(self.config['n_block']),
                                            k_quant_scheme=self.k_quant_scheme,
                                            v_quant_scheme=self.v_quant_scheme,
                                            residual_length=self.residual_length,
                                            sink=self.attn_sink,
                                            packing=self.packing,
                                            quant_kv_output=self.quant_kv_output,
                                            k_pruning_dim=self.k_pruning_dim,
                                            v_pruning_dim=self.v_pruning_dim)

        kv_methods = self.method['kv'] if isinstance(self.method.get('kv'), list) else [self.method.get('kv')]
        active_kv = 'hqq' if 'hqq' in kv_methods else 'kivi' if ('kivi' in kv_methods or 'think' in kv_methods) else 'fp16'

        if active_kv == 'hqq':
            self.model.generation_config.cache_config['k_bits'] = [x[0] for x in q_arch['k']]
            self.model.generation_config.cache_config['k_group_size'] = [x[1] for x in q_arch['k']]
            self.model.generation_config.cache_config['v_bits'] = [x[0] for x in q_arch['v']]
            self.model.generation_config.cache_config['v_group_size'] = [x[1] for x in q_arch['v']]
        elif active_kv == 'kivi':
            self.model.config.kivi_config.k_bits = [x[0] for x in q_arch['k']]
            self.model.config.kivi_config.k_group_size = [x[1] for x in q_arch['k']]
            self.model.config.kivi_config.v_bits = [x[0] for x in q_arch['v']]
            self.model.config.kivi_config.v_group_size = [x[1] for x in q_arch['v']]
            if hasattr(self.model.config, 'kivi_config'):
                if 'k' in p_arch:
                    self.model.config.kivi_config.k_pruning_dim = [int(d) for d in p_arch['k']]
                if 'v' in p_arch:
                    self.model.config.kivi_config.v_pruning_dim = [int(d) for d in p_arch['v']]
                # Enable ThinK whenever EITHER K or V has a non-zero pruning dim.
                # (Previously checked only K, so a V-only-pruned arch ran with
                # ThinK off -> the V pruning was a silent no-op while eff_kvbits
                # still credited the discount -> phantom "free" archs dominated
                # the high-eff_kvbits Pareto corner. See tests/test_vprune_gate.py.)
                self.model.config.kivi_config.enable_think = (
                    ('think' in kv_methods)
                    or any(int(d) > 0 for d in p_arch.get('k', [0]))
                    or any(int(d) > 0 for d in p_arch.get('v', [0]))
                )
        elif active_kv == 'fp16':
            pass
        else:
            raise NotImplementedError(self.method.get('kv'))

        return self.model

    def eval(self, accelerator, arch, metric, model=None, loss_func='cross_entropy', stride=0, prefill_prompt=False):
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
                stride=stride,
                last_tokens=self.last_tokens,
                prefill_prompt=prefill_prompt,
                dense_logits_list=self.dense_logits[dataset] if (self.loss_func in ['jsd', 'kld', 'topk', 'forward_kl']) else None,
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
        
        complexity = get_net_info(arch, self.config, self.group_size, n_token=self.n_token, attn_sink=self.attn_sink)
        return metric_list, complexity
    
    def remove_linears(self, model, config):
        for blk in getblock(model, config):
            for linear_group in config['linear']:
                for linear in linear_group.split(','):
                    delsubattr(blk, linear)
        clean_up()
