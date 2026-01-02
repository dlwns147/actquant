import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from .loss import JSD, compute_longppl, find_key_token, load_key_token, cal_overlap, compute_offsets


@torch.no_grad()
def eval_longppl(
    model, 
    accelerator, 
    loader, 
    evaluator_model=None,
    evaluator_tokenizer=None,
    tokenizer=None,
    seqlen=2048, 
    loss_func='longppl',  # 'longppl' or 'longppl_jsd'
    dense_logits_list=None, 
    key_token_list=None, 
    ignore_index=-100,
    trunc_len=4096,
    sliding_window=1024,
    alpha=2.0,
    beta=-2.0,
    save_path=None,
    mode='offline',  # 'online' or 'offline'
    evaluator_name="Meta-Llama-3.1-8B"
):
    """
    Evaluate LongPPL or LongPPL-JSD on a model using a data loader.
    
    Parameters:
        model: The model to evaluate
        accelerator: Accelerator object for distributed training
        loader: DataLoader containing input data
        evaluator_model: Model used to identify key tokens (for online mode)
        evaluator_tokenizer: Tokenizer for evaluator model
        tokenizer: Tokenizer for the evaluated model
        seqlen: Sequence length (not used directly, kept for compatibility)
        loss_func: 'longppl' for standard LongPPL, 'longppl_jsd' for JSD version
        dense_logits_list: List of dense logits for JSD calculation (required for 'longppl_jsd')
        key_token_list: Pre-computed key token list (optional)
        ignore_index: Index to ignore in loss calculation
        trunc_len: Length of truncated short context for LongPPL
        sliding_window: Size of sliding window for LongPPL
        alpha: Threshold for LSD in key token detection
        beta: Threshold for LCL in key token detection
        save_path: Path to save key tokens (optional)
        mode: 'online' to compute key tokens, 'offline' to use precomputed
        evaluator_name: Name of evaluator model for saving key tokens
    
    Returns:
        Dictionary with 'longppl' and 'ppl' scores
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided for eval_longppl")
    
    if loss_func == 'longppl_jsd' and dense_logits_list is None:
        raise ValueError("dense_logits_list must be provided for longppl_jsd")
    
    if mode == 'online' and evaluator_model is None:
        raise ValueError("evaluator_model must be provided for online mode")
    
    if mode == 'online' and evaluator_tokenizer is None:
        raise ValueError("evaluator_tokenizer must be provided for online mode")
    
    # Lists to store results
    longppls = []
    ppls = []
    nums_key_token = []
    nums_token = []
    
    # Create save directory if needed
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Process each batch
    for batch_idx, (inputs, attention_mask, labels) in enumerate(tqdm(loader, desc='Eval LongPPL')):
        # Decode each sequence in the batch
        batch_texts = []
        batch_key_tokens = []
        
        for seq_idx in range(inputs.shape[0]):
            # Decode the input sequence
            input_ids = inputs[seq_idx:seq_idx+1]
            # Remove padding tokens
            if attention_mask is not None:
                mask = attention_mask[seq_idx]
                actual_length = mask.sum().item()
                input_ids = input_ids[:, :actual_length]
            
            # Decode to text
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            batch_texts.append(text)
            
            # Get key tokens for this sequence if provided
            if key_token_list is not None and batch_idx < len(key_token_list):
                if seq_idx < len(key_token_list[batch_idx]):
                    batch_key_tokens.append(key_token_list[batch_idx][seq_idx])
                else:
                    batch_key_tokens.append(None)
            else:
                batch_key_tokens.append(None)
        
        # Process each text sequence
        for seq_idx, text in enumerate(batch_texts):
            # Determine save path for this sequence
            if save_path is not None:
                slice_save_path = os.path.join(save_path, f"batch_{batch_idx}_seq_{seq_idx}.txt")
            else:
                slice_save_path = None
            
            # Compute LongPPL
            if loss_func == 'longppl':
                # Standard LongPPL
                output = compute_longppl(
                    text=text,
                    model=model,
                    evaluator_model=evaluator_model,
                    tokenizer=tokenizer,
                    evaluator_tokenizer=evaluator_tokenizer,
                    save_path=slice_save_path,
                    trunc_len=trunc_len,
                    sliding_window=sliding_window,
                    alpha=alpha,
                    beta=beta
                )
                
                longppl = output['longppl']
                ppl = output['ppl']
                n_key_token = output['n_key_token']
                n_token = output['n_token']
                
            elif loss_func == 'longppl_jsd':
                # LongPPL with JSD
                # First, get key tokens
                try:
                    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
                    input_ids_seq = encoded_input['input_ids'].to(model.device)
                    offset_mapping = encoded_input['offset_mapping'][0]
                except NotImplementedError:
                    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False)
                    input_ids_seq = encoded_input['input_ids'].to(model.device)
                    offset_mapping = compute_offsets(text, tokenizer, input_ids_seq)[0]
                
                # Get key text slices
                if evaluator_model is not None:
                    key_text_slices = find_key_token(
                        text, evaluator_model, evaluator_tokenizer, 
                        trunc_len, sliding_window, slice_save_path, alpha, beta
                    )
                else:
                    key_text_slices = load_key_token(slice_save_path) if slice_save_path else None
                
                key_tokens = cal_overlap(offset_mapping, key_text_slices)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = model(input_ids_seq)
                    lm_logits = outputs.logits
                    
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
                    shift_labels = input_ids_seq[:, 1:].reshape(-1)
                
                # Compute overall PPL
                loss_func_ce = nn.CrossEntropyLoss()
                loss_all = loss_func_ce(shift_logits, shift_labels)
                ppl = torch.exp(loss_all).item()
                n_token = input_ids_seq.shape[-1]
                
                # Compute JSD on key tokens if available
                if key_tokens is not None and len(key_tokens) > 0:
                    # Get dense logits for this sequence
                    dense_logits_seq = None
                    if dense_logits_list is not None and batch_idx < len(dense_logits_list):
                        dense_logits = dense_logits_list[batch_idx]
                        if isinstance(dense_logits, list) and seq_idx < len(dense_logits):
                            dense_logits_seq = dense_logits[seq_idx]
                        elif isinstance(dense_logits, torch.Tensor):
                            # Assume batch dimension is first
                            if len(dense_logits.shape) == 3 and dense_logits.shape[0] > seq_idx:
                                dense_logits_seq = dense_logits[seq_idx]
                    
                    if dense_logits_seq is not None:
                        # Handle shape: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
                        if len(dense_logits_seq.shape) == 3:
                            dense_logits_seq = dense_logits_seq[0]
                        # Shift to match shift_logits
                        dense_logits_seq = dense_logits_seq[:-1, :].contiguous()
                        dense_logits_seq = dense_logits_seq.reshape(-1, dense_logits_seq.size(-1))
                        
                        # Compute JSD on key tokens
                        jsd_loss = JSD()
                        jsd_key = jsd_loss(
                            shift_logits[key_tokens], 
                            dense_logits_seq[key_tokens]
                        )
                        # Convert JSD to perplexity-like metric (exp of JSD)
                        longppl = torch.exp(jsd_key).item()
                        n_key_token = len(key_tokens)
                    else:
                        # Fallback to standard cross-entropy on key tokens if no dense logits
                        loss_key = loss_func_ce(shift_logits[key_tokens], shift_labels[key_tokens])
                        longppl = torch.exp(loss_key).item()
                        n_key_token = len(key_tokens)
                else:
                    # No key tokens
                    longppl = None
                    n_key_token = 0
            else:
                raise NotImplementedError(f"loss_func '{loss_func}' is not implemented for eval_longppl")
            
            # Accumulate results
            if longppl is not None:
                longppls.append(longppl)
                nums_key_token.append(n_key_token)
            ppls.append(ppl)
            nums_token.append(n_token)
    
    # Aggregate results across all sequences
    if len(longppls) > 0 and len(nums_key_token) > 0:
        # Weighted average of log perplexities
        log_longppls = np.log(np.array(longppls))
        weights_key = np.array(nums_key_token)
        longppl = np.exp((log_longppls * weights_key).sum() / weights_key.sum())
    else:
        longppl = None
    
    if len(ppls) > 0 and len(nums_token) > 0:
        log_ppls = np.log(np.array(ppls))
        weights_token = np.array(nums_token)
        ppl = np.exp((log_ppls * weights_token).sum() / weights_token.sum())
    else:
        ppl = None
    
    return {"longppl": longppl, "ppl": ppl}


def eval_metric_longppl(
    model, 
    accelerator, 
    metric, 
    loader, 
    seqlen, 
    loss_func='longppl',
    dense_logits_list=None, 
    key_token_list=None,
    evaluator_model=None,
    evaluator_tokenizer=None,
    tokenizer=None,
    trunc_len=4096,
    sliding_window=1024,
    alpha=2.0,
    beta=-2.0,
    save_path=None,
    mode='offline',
    evaluator_name="Meta-Llama-3.1-8B",
    **kwargs
):
    """
    Wrapper function to evaluate LongPPL metrics.
    Similar to eval_metric but for LongPPL.
    """
    if metric == 'longppl' or metric == 'longppl_jsd':
        return eval_longppl(
            model=model,
            accelerator=accelerator,
            loader=loader,
            evaluator_model=evaluator_model,
            evaluator_tokenizer=evaluator_tokenizer,
            tokenizer=tokenizer,
            seqlen=seqlen,
            loss_func=metric,
            dense_logits_list=dense_logits_list,
            key_token_list=key_token_list,
            trunc_len=trunc_len,
            sliding_window=sliding_window,
            alpha=alpha,
            beta=beta,
            save_path=save_path,
            mode=mode,
            evaluator_name=evaluator_name
        )
    else:
        raise NotImplementedError(f"metric '{metric}' is not supported for eval_metric_longppl")

