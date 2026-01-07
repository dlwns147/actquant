import argparse
import os
import json
import torch
import warnings
warnings.simplefilter("ignore")

from accelerate import Accelerator
from utils.func import init_accelerator, set_seed, get_hfmodel, clean_up
from utils.data import get_loader, get_tokenizer
from utils.loss import get_key_token_list

def main(args):
    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    # Initialize accelerator
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    
    # Load model and tokenizer
    model_id = f'{args.model_path}/{args.model_name}' if args.model_path else args.model_name
    accelerator.print(f"Loading model from: {model_id}")
    
    model = get_hfmodel(
        model_id, 
        dtype=args.dtype, 
        device_map=device_map
    )
    tokenizer = get_tokenizer(model_id, use_fast=True)
    # tokenizer = get_tokenizer(model_id)
    
    # Create data loader
    accelerator.print(f"Creating data loader for dataset: {args.dataset}")
    loader = get_loader(
        args.dataset,
        model=model_id,
        n_sample=args.n_sample,
        batch_size=args.data_batch_size,
        # train=True,
        train=False,
        seed=args.seed,
        seqlen=args.seqlen,
        min_seqlen=args.min_seqlen
    )
    loader = accelerator.prepare(loader)
    
    # Prepare save path
    if args.save_path:
        save_path = args.save_path
    else:
        # Generate default save path based on parameters
        save_path = f"key_token/{args.model_name}_{args.n_sample}sample_{args.seqlen}seqlen_{args.min_seqlen}min_{args.trunc_len}trunc_{args.sliding_window}sw_{args.alpha}alpha_{args.beta}beta"
    
    # Create directory for each dataset
    dataset_save_path = os.path.join(save_path, args.dataset)
    os.makedirs(dataset_save_path, exist_ok=True)
    
    accelerator.print(f"Saving key tokens to: {dataset_save_path}")
    
    # Generate key token list
    accelerator.print("Generating key token list...")
    key_token_list = get_key_token_list(
        evaluator_model=model,
        evaluator_tokenizer=tokenizer,
        loader=loader,
        tokenizer=tokenizer,
        trunc_len=args.trunc_len,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        beta=args.beta,
        save_path=dataset_save_path,
        mode='online',
        verbosity=args.verbosity
    )
    
    # Count total key tokens
    n_key_token = sum([len(key_token) if key_token is not None else 0 for key_token in key_token_list])
    n_key_token = sum(accelerator.gather_for_metrics([n_key_token], use_gather_object=True))
    
    accelerator.print(f'Dataset: {args.dataset}, Total key tokens: {n_key_token}')
    
    # Decode key tokens back to text and print
    accelerator.print("Decoding key tokens to text...")
    bs = args.data_batch_size
    for batch_idx, (inputs, attention_mask, labels) in enumerate(loader):
        if batch_idx >= len(key_token_list):
            break

        batch_key_tokens = key_token_list[batch_idx * bs:(batch_idx + 1) * bs]
        for seq_idx in range(bs):
            slice_idx = batch_idx * bs + seq_idx
            key_tokens = batch_key_tokens[seq_idx]
            if key_tokens is None or len(key_tokens) == 0:
                continue

            input_ids = inputs[seq_idx]
            if attention_mask is not None:
                mask = attention_mask[seq_idx]
                actual_length = mask.sum().item()
                input_ids = input_ids[:actual_length]

            # key_tokens are indices for shift_logits (predicting token at idx+1)
            # So actual input_ids index is idx + 1
            # Filter valid indices and convert to input_ids indices
            key_token_ids = []
            for kt in key_tokens:
                actual_idx = kt + 1  # Convert from shift_logits index to input_ids index
                if 0 <= actual_idx < input_ids.shape[0]:
                    key_token_ids.append(input_ids[actual_idx].item())
            
            if len(key_token_ids) == 0:
                continue

            key_text = tokenizer.decode(key_token_ids, skip_special_tokens=True)

            # accelerator.print(f"[batch {batch_idx} seq {seq_idx}] key token indices (shift_logits): {key_tokens}")
            # accelerator.print(f"[batch {batch_idx} seq {seq_idx}] key token indices (input_ids): {valid_indices}")
            accelerator.print(f"[Slice {slice_idx}] key token text: {key_text if len(key_text) < 200 else key_text[:200] + '...'}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate key token list from model')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to model directory')
    parser.add_argument('--model_name', type=str, default='',
                        help='model name (e.g., Llama-3.1-8B-Instruct)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset name (wikitext2, c4, gsm8k, etc.)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--config', type=str, default='',
                        help='path to config json file')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='number of samples to process')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='batch size for data loader')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequence length')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequence length')
    parser.add_argument('--trunc_len', type=int, default=512,
                        help='truncation length for long PPL/JSD calculation')
    parser.add_argument('--sliding_window', type=int, default=128,
                        help='sliding_window length for long PPL/JSD calculation')
    parser.add_argument('--alpha', type=int, default=2,
                        help='Long-short distance (LSD) threshold for long PPL/JSD calculation')
    parser.add_argument('--beta', type=int, default=-2,
                        help='Long context likelihood (LCL) threshold for long PPL/JSD calculation')
    parser.add_argument('--save_path', type=str, default='',
                        help='path to save key tokens (default: auto-generated)')
    # parser.add_argument('--load_path', type=str, default='',
    #                     help='path to load precomputed key tokens (for offline mode)')
    # parser.add_argument('--save_list', action='store_true',
    #                     help='save key_token_list as pickle file')
    parser.add_argument('--dtype', type=str, default='auto',
                        help='model dtype (auto, float16, bfloat16, etc.)')
    parser.add_argument('--verbosity', action='store_true',
                        help='')
    
    args = parser.parse_args()
    main(args)
