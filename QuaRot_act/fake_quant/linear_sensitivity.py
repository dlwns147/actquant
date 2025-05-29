import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils
import csv
import gc
import time

def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    print(f'utils.DEV : {utils.DEV}')
    

    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

        
    n_layer = len(model_utils.get_layers(model))
    arch = {
        'w': {
            'self_attn.q_proj': [args.w_bits] * n_layer,
            'self_attn.k_proj': [args.w_bits] * n_layer,
            'self_attn.v_proj': [args.w_bits] * n_layer,
            'self_attn.o_proj': [args.w_bits] * n_layer,
            'mlp.gate_proj': [args.w_bits] * n_layer,
            'mlp.up_proj': [args.w_bits] * n_layer,
            'mlp.down_proj': [args.w_bits] * n_layer,
        },
        'a': {
            # 'self_attn.q_proj,self_attn.k_proj,self_attn.v_proj': [args.a_bits] * n_layer,
            # 'self_attn.o_proj': [args.a_bits] * n_layer,
            # 'mlp.gate_proj,mlp.up_proj': [args.a_bits] * n_layer,
            # 'mlp.down_proj': [args.a_bits] * n_layer,
            'qkv': [args.a_bits] * n_layer,
            'o': [args.a_bits] * n_layer,
            'gateup': [args.a_bits] * n_layer,
            'down': [args.a_bits] * n_layer,
        },
        'v': [args.v_bits] * n_layer,
        'k': [args.k_bits] * n_layer,
    }
    linear_hierarchy = {
        'self_attn.q_proj': 'qkv',
        'self_attn.k_proj': 'qkv',
        'self_attn.v_proj': 'qkv',
        'self_attn.o_proj': 'o',
        'mlp.gate_proj': 'gateup',
        'mlp.up_proj': 'gateup',
        'mlp.down_proj': 'down',
    }

    linear_list = []
    ppl_list = {d: [] for d in args.eval_datasets}
    loss_list =  []
    
    loss_loader = data_utils.get_train_loaders(args.loss_dataset, seed=args.seed, nsamples=args.nsamples, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token)
    ppl_loaders = {dataset: data_utils.get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token, eval_mode=True) for dataset in args.eval_datasets} if args.eval_ppl else None
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    dense_logits_list = eval_utils.get_dense_logits(model, loss_loader, device=utils.DEV) if args.loss_func == 'jsd' else None
    model.to('cpu')

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    start_point = time.time()
    for i in range(n_layer):
        if args.sensitivity_module == 'a':
            loop_list = arch['a'].keys()
        elif args.sensitivity_module in ['k', 'v']:
            loop_list = range(1)
            
        # for cur_linear_group in arch['a'].keys():
        for cur_linear_group in loop_list:
            iter_start = time.time()

            if args.sensitivity_module == 'a':
                key = f'{i}.{cur_linear_group}'
                linear_list.append(key)
                cur_bits = arch['a'][cur_linear_group][i]
                arch['a'][cur_linear_group][i] = args.min_bits

            elif args.sensitivity_module in ['k', 'v']:
                linear_list.append(i)
                cur_bits = arch[args.sensitivity_module][i]
                arch[args.sensitivity_module][i] = args.min_bits
                key = f'{i}.{args.sensitivity_module}'

            # Add Input Quantization
            if args.a_bits < 16 or args.v_bits < 16:
                qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
                down_proj_groupsize = -1
                if args.a_groupsize > 0 and "llama" in args.model:
                    down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
               
                for name in qlayers:
                    # print(f'name : {name}')
                    if 'lm_head' not in name:
                        _, layer_idx, module, linear_name = name.rsplit('.', maxsplit=3)

                        # layer_input_bits = arch['a'][f'{module}.{linear_name}'][int(layer_idx)] # args.a_bits 
                        layer_input_bits = arch['a'][linear_hierarchy[f'{module}.{linear_name}']][int(layer_idx)] # args.a_bits 
                    layer_groupsize = args.a_groupsize
                    layer_a_sym = not(args.a_asym)
                    layer_a_clip = args.a_clip_ratio
                    
                    if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                        qlayers[name].out_quantizer.configure(bits=arch['v'][int(layer_idx)],
                                                    groupsize=args.v_groupsize,
                                                    sym=not(args.v_asym),
                                                    clip_ratio=args.v_clip_ratio)
                    
                    if 'lm_head' in name: #Skip lm_head quantization   
                        layer_input_bits = 16
                    
                    if 'down_proj' in name: #Set the down_proj precision
                        if args.int8_down_proj:
                            layer_input_bits = 8
                        layer_groupsize = down_proj_groupsize

                        
                    qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                    groupsize=layer_groupsize,
                                                    sym=layer_a_sym,
                                                    clip_ratio=layer_a_clip)

                if args.k_bits < 16:
                    if args.k_pre_rope:
                        raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
                    else:
                        rope_function_name = model_utils.get_rope_function_name(model)
                        layers = model_utils.get_layers(model)
                        for layer_idx, layer in enumerate(layers):    
                            k_quant_config = {'k_bits':arch['k'][int(layer_idx)], "k_groupsize": args.k_groupsize,
                                                        "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}

                            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                                        layer.self_attn, 
                                        rope_function_name, 
                                        config=model.config,
                                        **k_quant_config)
                    
                # Evaluating on dataset
                loss = eval_utils.get_loss(model, loss_loader, loss_func=args.loss_func, dense_logits_list=dense_logits_list, device=utils.DEV)
                loss_list.append(loss)
                if args.wandb:
                        wandb.log({'loss/{}'.format(args.loss_dataset.upper()): loss})
                
                if args.eval_ppl:
                    for dataset in args.eval_datasets:
                        testloader = ppl_loaders[dataset]
                        # dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, dataset, args)
                        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, dataset, args, layerwise=args.eval_layerwise)
                        if args.wandb:
                                wandb.log({'ppl/{}'.format(dataset.upper()): dataset_ppl})
                        ppl_list[dataset].append(dataset_ppl)
                iter_time = time.time() - iter_start
                print(f"[{key} replaced] Loss={loss:.4f}, PPL: {[p[-1] for p in ppl_list] if args.eval_ppl else 0}, time: {iter_time:.2f}")

            if args.sensitivity_module == 'a':
                arch['a'][cur_linear_group][i] = cur_bits
            elif args.sensitivity_module in ['k', 'v']:
                arch[args.sensitivity_module][i] = cur_bits

            if args.sensitivity_file:
                with open(args.sensitivity_file, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(linear_list)
                    write.writerow(loss_list)
            
            if args.eval_ppl and args.sensitivity_ppl_file:
                    write = csv.writer(f)
                    write.writerow(linear_list)
                    if args.eval_ppl:
                        for ppl in ppl_list.values():
                            write.writerow(ppl)

    finish_point = time.time()
    time_elapsed = finish_point - start_point

    print(linear_list)
    print(ppl_list)
    print(f"Time_Elapsed: {time_elapsed}")
    print(args)



if __name__ == '__main__':
    main()
