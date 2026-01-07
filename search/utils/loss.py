import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os

# class JSD(nn.Module):
#     def __init__(self, reduction='batchmean'):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)

#     def forward(self, p: torch.tensor, q: torch.tensor):
#         p, q = p.log_softmax(-1), q.log_softmax(-1)
#         m = (0.5 * (p + q))
#         return 0.5 * (self.kl(m, p) + self.kl(m, q))

class JSD(nn.Module):
    def __init__(self, reduction='batchmean', eps=1e-7):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.eps = eps

    def forward(self, p: torch.tensor, q: torch.tensor):
        m = (0.5 * (p.softmax(-1) + q.softmax(-1))).clamp_min(self.eps).log()
        return 0.5 * (self.kl(m, p.log_softmax(-1)) + self.kl(m, q.log_softmax(-1)))

def TopK(p: torch.tensor, q: torch.tensor, k: int):
    p_topk, q_topk = p.topk(k, dim=-1, largest=True), q.topk(k, dim=-1, largest=True)
    pq = torch.cat((p_topk, q_topk), dim=-1)
    union, counts = pq.unique(dim=-1, return_inverse=False, return_counts=True)
    intersection = pq[torch.where(counts.gt(1))]
    return (intersection / union).mean()

def get_key_token_list(
    evaluator_model, 
    evaluator_tokenizer, 
    loader, 
    tokenizer=None,
    save_path='', 
    load_path='', 
    trunc_len=4096, 
    sliding_window=1024, 
    alpha=2, 
    beta=-2,
    mode='offline',
    verbosity=False
):
    """
    Get key token list from loader.
    
    Parameters:
        evaluator_model: Model used to identify key tokens (for online mode)
        evaluator_tokenizer: Tokenizer for evaluator model
        loader: DataLoader containing input data
        tokenizer: Tokenizer for the evaluated model (optional, for decode if needed)
        save_path: Path to save key tokens (for online mode)
        load_path: Path to load precomputed key tokens (for offline mode)
        trunc_len: Length of truncated short context
        sliding_window: Size of sliding window
        alpha: Threshold for LSD
        beta: Threshold for LCL
        mode: 'online' to compute key tokens, 'offline' to use precomputed
        verbosity: If True, print decoded key tokens for debugging
    
    Returns:
        List of key token indices per batch: [batch_idx][seq_idx] -> list of token indices
    """
    key_token_list = []  
    if tokenizer is None:
        tokenizer = evaluator_tokenizer
    for batch_idx, (inputs, attention_mask, labels) in enumerate(loader):
        batch_key_tokens = []
        batch_size = inputs.shape[0]
        
        for seq_idx in range(batch_size):
            # Get actual input_ids (remove padding)
            slice_idx = batch_idx * batch_size + seq_idx
            input_ids = inputs[seq_idx:seq_idx+1]
            if attention_mask is not None:
                mask = attention_mask[seq_idx]
                actual_length = mask.sum().item()
                input_ids = input_ids[:, :actual_length]
            
            # For offline mode, try to load from file
            if mode == 'offline':
                # slice_path = os.path.join(load_path, f"batch_{batch_idx}_seq_{seq_idx}.txt")\
                assert os.path.exists(load_path)
                slice_path = os.path.join(load_path, f"slice_{slice_idx}.txt")
                assert os.path.exists(slice_path)
                # Need to decode to get text for offset mapping
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                try:
                    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
                    offset_mapping = encoded_input['offset_mapping'][0]
                except NotImplementedError:
                    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False)
                    offset_mapping = compute_offsets(text, tokenizer, input_ids)[0]
                
                key_text_slices = load_key_token(slice_path)
                if key_text_slices is not None:
                    key_tokens = cal_overlap(offset_mapping, key_text_slices)
                    batch_key_tokens.append(key_tokens)
                    
                    # Print decoded key tokens if verbosity is enabled
                    if verbosity and key_tokens is not None and len(key_tokens) > 0:
                        # key_tokens are indices for shift_logits (predicting token at idx+1)
                        # So actual input_ids index is idx + 1
                        key_token_ids = [input_ids[0, idx + 1].item() for idx in key_tokens]                            
                        if key_token_ids:
                            decoded_tokens = tokenizer.decode(key_token_ids, skip_special_tokens=True)
                            print(f"[Offline] [Slice {slice_idx}] {len(key_tokens)} key tokens: {decoded_tokens[:200]}")
                        else:
                            print(f"[Offline] [Slice {slice_idx}] {len(key_tokens)} key tokens (could not decode)")
                        
                else:
                    batch_key_tokens.append(None)

            elif mode == 'online':
                assert evaluator_model is not None
                # Need to decode for online mode
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                
                # slice_save_path = os.path.join(save_path, f"batch_{batch_idx}_seq_{seq_idx}.txt") if save_path else ''
                slice_save_path = os.path.join(save_path, f"slice_{slice_idx}.txt") if save_path else ''
                key_text_slices = find_key_token(
                    text, evaluator_model, evaluator_tokenizer, 
                    trunc_len, sliding_window, slice_save_path, alpha, beta
                )
                
                try:
                    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
                    offset_mapping = encoded_input['offset_mapping'][0]
                except NotImplementedError:
                    encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False)
                    offset_mapping = compute_offsets(text, tokenizer, input_ids)[0]
                
                if key_text_slices is not None:
                    key_tokens = cal_overlap(offset_mapping, key_text_slices)
                    batch_key_tokens.append(key_tokens)
                    
                    # Print decoded key tokens if verbosity is enabled
                    if verbosity and key_tokens is not None and len(key_tokens) > 0:
                        # key_tokens are indices for shift_logits (predicting token at idx+1)
                        # So actual input_ids index is idx + 1
                        key_token_ids = [input_ids[0, idx + 1].item() for idx in key_tokens]                             
                        if key_token_ids:
                            decoded_tokens = tokenizer.decode(key_token_ids, skip_special_tokens=True)
                            print(f"[Online] [Slice {slice_idx}] {len(key_tokens)} key tokens: {decoded_tokens[:200]}")
                        else:
                            print(f"[Online] [Slice {slice_idx}] {len(key_tokens)} key tokens (could not decode)")
                else:
                    batch_key_tokens.append(None)
            else:
                raise NotImplementedError
        
        key_token_list.extend(batch_key_tokens)
    
    return key_token_list
    

def merge_intervals(intervals):
    if intervals.size(0) == 0:
        return intervals

    start = intervals[:, 0]
    end = intervals[:, 1]
    adjacent = (start[1:] - end[:-1]) == 0

    keep_start_mask = torch.cat([torch.tensor([True]), ~adjacent])
    merged_start = start[keep_start_mask]
    keep_end_mask = torch.cat([~adjacent, torch.tensor([True])])
    merged_end = end[keep_end_mask]

    merged_intervals = torch.stack([merged_start, merged_end], dim=1)
    
    return merged_intervals 

def find_key_token(text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path='', alpha=2, beta=-2):
    text_encoded = evaluator_tokenizer(text, return_tensors="pt", return_offsets_mapping=True)               
    input_ids = text_encoded['input_ids'].to(evaluator_model.device)
    
    with torch.no_grad():
        output_full = evaluator_model(input_ids)
    shift_full_logits = output_full.logits
    # shift_full_logits = output_full.logits[:, :-1, :].contiguous()
    # shift_full_logits = shift_full_logits.reshape(-1, shift_full_logits.size(-1))
    
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    bs, max_len = input_ids.shape
    key_tokens = []

    with torch.no_grad():
        for i, start_token in enumerate(range(0, max_len-trunc_len, sliding_window)):
            if start_token+trunc_len+sliding_window > max_len:
                sliding_window = max_len-start_token-trunc_len

            input_ids_short = input_ids[:, start_token: start_token+trunc_len+sliding_window]
            output_short = evaluator_model(input_ids_short)
            shift_short_logits = output_short.logits[:, trunc_len-1: trunc_len+sliding_window-1, :].contiguous()
            shift_short_logits = shift_short_logits.reshape(-1, shift_short_logits.size(-1))
            shift_short_labels = input_ids_short[:, trunc_len: trunc_len+sliding_window].reshape(-1)
            
            shift_full_trunc_logits = shift_full_logits[:, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :].reshape(-1, shift_full_logits.size(-1))
            shift_full_labels = input_ids[:, start_token+trunc_len: start_token+trunc_len+sliding_window].reshape(-1)

            loss_full = loss_f(shift_full_trunc_logits, shift_full_labels)
            loss_short = loss_f(shift_short_logits, shift_short_labels)

            # loss_full = loss_f(output_full.logits[0, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :], input_ids[0, start_token+trunc_len: start_token+trunc_len+sliding_window])
            # loss_short = loss_f(output_short.logits[0, trunc_len-1: trunc_len+sliding_window-1, :], input_ids_short[0, trunc_len: trunc_len+sliding_window])

            # loss_discrepancy = (torch.logical_and((loss_short - loss_full) > alpha, loss_full < (beta * -1))).squeeze()
            loss_discrepancy = (torch.logical_and((loss_short - loss_full) > alpha, loss_full < (beta * -1))).flatten()

            for i, is_key in enumerate(loss_discrepancy):
                if is_key:
                    key_tokens.append(start_token+trunc_len+i)
    
    # key_text_intervals = merge_intervals(text_encoded['offset_mapping'][0, key_tokens])
    # key_text_intervals = merge_intervals(text_encoded['offset_mapping'].reshape(-1, 2)[key_tokens])
    key_text_intervals = merge_intervals(text_encoded['offset_mapping'].squeeze(0)[key_tokens])

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            slices_str = ";".join([f"[{element[0]}, {element[1]}]" for element in key_text_intervals])
            f.write(slices_str)

    return key_text_intervals

def load_key_token(save_path):
    with open(save_path, "r+", encoding="utf-8") as f:
        for line in f.readlines():
            key_slices_str = line.split(';')
            key_text_slices = []
            for key_slice in key_slices_str:
                key_text_slices.append(eval(key_slice))
            return key_text_slices

def cal_overlap(offset_mapping, key_text_slices):
    if key_text_slices is None:
        return None

    key_tokens = []
    i, j = 0, 0
    
    while i < len(offset_mapping) and j < len(key_text_slices):
        a_start, a_end = offset_mapping[i]
        b_start, b_end = key_text_slices[j]

        if a_start >= b_start and a_end <= b_end:
            key_tokens.append(i-1)
            i += 1
        elif a_start < b_start:
            i += 1
        else:
            j += 1

    return key_tokens

def compute_offsets(text, tokenizer, input_ids):
    """
    Compute character-level offset mappings for tokens when tokenizer doesn't support return_offsets_mapping.
    
    Parameters:
        text: Original text string
        tokenizer: Tokenizer instance
        input_ids: Tensor of shape [batch_size, seq_len] or [1, seq_len]
    
    Returns:
        Tensor of shape [batch_size, seq_len, 2] with [start, end] offsets for each token
    """
    batch_size, seq_len = input_ids.shape
    total_offsets = []
    
    for batch_idx in range(batch_size):
        offsets = []
        text_pointer = 0
        
        for token_idx in range(seq_len):
            token_id = input_ids[batch_idx, token_idx].item()
            
            if token_idx == 0:
                # First token: decode it to get its length
                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                token_len = len(token_text)
            else:
                # Subsequent tokens: decode cumulative to get incremental length
                prev_token_id = input_ids[batch_idx, token_idx - 1].item()
                cumulative_text = tokenizer.decode([prev_token_id, token_id], skip_special_tokens=True)
                prev_text = tokenizer.decode([prev_token_id], skip_special_tokens=True)
                token_len = len(cumulative_text) - len(prev_text)
            
            offsets.append([text_pointer, text_pointer + token_len])
            text_pointer += token_len
        
        total_offsets.append(offsets)
    
    # Return as tensor: [batch_size, seq_len, 2]
    return torch.tensor(total_offsets)


# def compute_longppl(
#         text,
#         model,
#         evaluator_model=None,
#         tokenizer=None, 
#         evaluator_tokenizer=None, 
#         save_path='', 
#         load_path='',
#         key_token_list=None,
#         loss_func='longppl',  # 'longppl' or 'longjsd'
#         dense_logits_list=None,  # Required for 'longjsd'
#         trunc_len=4096, 
#         sliding_window=1024,
#         alpha=2,
#         beta=-2
#     ):
#     r"""
#     Compute the LongPPL or LongJSD for long text sequences.

#     Parameters:
#         text (`str` or `list`): 
#             The input text(s) for which LongPPL/LongJSD is calculated.
#         model (`transformers.PretrainedModel`): 
#             The model used for LongPPL/LongJSD calculation.
#         evaluator_model (`transformers.PretrainedModel`, *optional*): 
#             The evaluator model used to identify the key tokens (for online mode).
#         tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
#             Tokenizer of the evaluated model.
#         evaluator_tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
#             Tokenizer of the evaluator model (for online mode).
#         save_path (`str`, *optional*): If specified, the path to save the computed key tokens.
#         load_path (`str`, *optional*): If specified, the path to load precomputed key tokens (for offline mode).
#         key_token_list (`list`, *optional*): Pre-computed key token indices list. If provided, this takes priority.
#         loss_func (`str`, *optional*, default='longppl`): 'longppl' for standard LongPPL, 'longjsd' for JSD version.
#         dense_logits_list (`list` or `torch.Tensor`, *optional*): Dense logits for JSD calculation (required for 'longjsd').
#         trunc_len (`int`, *optional*, default=4096): Length of the truncated short context.
#         sliding_window (`int`, *optional*, default=1024): Number of tokens sharing the same short context.
#         alpha (`float`, *optional*, default=2): Threshold for LSD in key token detection.
#         beta (`float`, *optional*, default=-2): Threshold for LCL in key token detection.

#     Returns:
#         [`Dict`]: A `Dict` object including:
#             - 'longppl' (`float`, *optional*): The LongPPL score (for 'longppl' mode).
#             - 'longjsd' (`float`, *optional*): The LongJSD score (for 'longjsd' mode).
#             - 'n_key_token' (`int`): The number of key tokens (under the evaluated model).
#             - 'ppl' (`float`): The PPL score.
#             - 'n_token' (`int`): The number of tokens in the input text.
#     """
#     if loss_func == 'longjsd' and dense_logits_list is None:
#         raise ValueError("dense_logits_list must be provided for longjsd")
    
#     assert type(text) in [str, list]
#     if type(text) == str:
#         text = [text]
#     total_seqlen = 0
#     total_key_token_len = 0
#     nll_all_list = []
#     nll_key_list = []
#     jsd_key_list = []  # Store JSD values for each sequence
#     jsd_key_token_counts = []  # Store key token counts for each sequence (for weighted average)
    
#     for text_idx, cur_text in enumerate[str](text):
#         try:
#             encoded_input = tokenizer(cur_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
#             input_ids = encoded_input['input_ids'].to(model.device)
#             offset_mapping = encoded_input['offset_mapping'][0]
#         except NotImplementedError:
#             encoded_input = tokenizer(cur_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False)
#             input_ids = encoded_input['input_ids'].to(model.device)
#             offset_mapping = compute_offsets(cur_text, tokenizer, input_ids)[0]
        
#         # Get key tokens with priority: key_token_list > load_path > evaluator_model
#         key_tokens = None
#         key_text_slices = None
        
#         if key_token_list is not None and text_idx < len(key_token_list):
#             # Use precomputed key token indices
#             key_tokens = key_token_list[text_idx]
#             if not isinstance(key_tokens, list):
#                 if isinstance(key_tokens, torch.Tensor):
#                     key_tokens = key_tokens.cpu().tolist()
#                 else:
#                     key_tokens = [key_tokens]
#         elif load_path and os.path.exists(load_path):
#             # Load from file (offline mode)
#             key_text_slices = load_key_token(load_path)
#             if key_text_slices is not None:
#                 key_tokens = cal_overlap(offset_mapping, key_text_slices)
#         elif evaluator_model is not None:
#             # Compute key tokens (online mode)
#             key_text_slices = find_key_token(cur_text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path, alpha, beta)
#             if key_text_slices is not None:
#                 key_tokens = cal_overlap(offset_mapping, key_text_slices)
        
#         bs, seqlen = input_ids.shape
#         key_token_len = len(key_tokens) if key_tokens is not None else 0
        
#         with torch.no_grad():
#             outputs = model(input_ids)
#         lm_logits = outputs.logits
            
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
#         shift_labels = input_ids[:, 1:].reshape(-1)
        
#         loss_func_ce = torch.nn.CrossEntropyLoss()
#         loss_all = loss_func_ce(shift_logits, shift_labels)
        
#         nll_all = loss_all.float() * seqlen * bs
#         nll_all_list.append(nll_all)
#         total_seqlen += seqlen * bs
        
#         if key_tokens is not None and len(key_tokens) > 0:
#             # Filter key_tokens to valid indices
#             valid_key_tokens = [kt for kt in key_tokens if 0 <= kt < shift_logits.shape[0]]
            
#             if len(valid_key_tokens) > 0:
#                 valid_key_tokens_tensor = torch.tensor(valid_key_tokens, device=shift_logits.device)
                
#                 if loss_func == 'longjsd':
#                     # Compute JSD on key tokens
#                     dense_logits_seq = None
#                     if dense_logits_list is not None:
#                         if isinstance(dense_logits_list, list) and text_idx < len(dense_logits_list):
#                             dense_logits_seq = dense_logits_list[text_idx]
#                         elif isinstance(dense_logits_list, torch.Tensor):
#                             # Handle different tensor shapes: [batch, seq, vocab] or [seq, vocab]
#                             if len(dense_logits_list.shape) == 3:
#                                 # [batch, seq, vocab]
#                                 if dense_logits_list.shape[0] > text_idx:
#                                     dense_logits_seq = dense_logits_list[text_idx]
#                             elif len(dense_logits_list.shape) == 2:
#                                 # [seq, vocab] - single sequence
#                                 if len(text) == 1:
#                                     dense_logits_seq = dense_logits_list
                    
#                     if dense_logits_seq is not None:
#                         # Handle shape: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
#                         if len(dense_logits_seq.shape) == 3:
#                             dense_logits_seq = dense_logits_seq[0]
#                         # Ensure device match
#                         if isinstance(dense_logits_seq, torch.Tensor):
#                             dense_logits_seq = dense_logits_seq.to(model.device)
#                         # Shift to match shift_logits
#                         dense_logits_seq = dense_logits_seq[:-1, :].contiguous()
#                         dense_logits_seq = dense_logits_seq.reshape(-1, dense_logits_seq.size(-1))
                        
#                         # Compute JSD on key tokens
#                         jsd_loss = JSD()
#                         jsd_key = jsd_loss(
#                             shift_logits[valid_key_tokens_tensor], 
#                             dense_logits_seq[valid_key_tokens_tensor]
#                         )
#                         # Store JSD value and token count for weighted average
#                         jsd_key_list.append(jsd_key.item())
#                         jsd_key_token_counts.append(len(valid_key_tokens) * bs)
#                         total_key_token_len += len(valid_key_tokens) * bs
#                     else:
#                         # Fallback to standard cross-entropy on key tokens if no dense logits
#                         loss_key = loss_func_ce(shift_logits[valid_key_tokens_tensor], shift_labels[valid_key_tokens_tensor])
#                         nll_key = loss_key.float() * len(valid_key_tokens) * bs
#                         nll_key_list.append(nll_key)
#                         total_key_token_len += len(valid_key_tokens) * bs
#                 else:
#                     # Standard LongPPL: use cross-entropy loss
#                     loss_key = loss_func_ce(shift_logits[valid_key_tokens_tensor], shift_labels[valid_key_tokens_tensor])
#                     nll_key = loss_key.float() * len(valid_key_tokens) * bs
#                     nll_key_list.append(nll_key)
#                     total_key_token_len += len(valid_key_tokens) * bs
    
#     ppl_all = torch.exp(sum(nll_all_list) / total_seqlen) if total_seqlen > 0 else None
    
#     result = {
#         "n_key_token": total_key_token_len,
#         "ppl": ppl_all,
#         "n_token": total_seqlen
#     }
    
#     if loss_func == 'longppl':
#         ppl_key = torch.exp(sum(nll_key_list) / total_key_token_len) if total_key_token_len > 0 else None
#         result["longppl"] = ppl_key
#     elif loss_func == 'longjsd':
#         if len(jsd_key_list) > 0 and len(jsd_key_token_counts) > 0:
#             # Weighted average of log JSD values
#             log_jsds = np.log(np.array(jsd_key_list))
#             weights_key = np.array(jsd_key_token_counts)
#             longjsd = np.exp((log_jsds * weights_key).sum() / weights_key.sum())
#             result["longjsd"] = longjsd
#         else:
#             # Fallback to cross-entropy if no JSD computed
#             if len(nll_key_list) > 0 and total_key_token_len > 0:
#                 ppl_key = torch.exp(sum(nll_key_list) / total_key_token_len)
#                 result["longjsd"] = ppl_key.item()
#             else:
#                 result["longjsd"] = None
    
#     return result

    
        
        # loss_f = torch.nn.CrossEntropyLoss(reduction='none')
        # loss_overall = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float).cpu().numpy()
        
        # if key_tokens is None or len(key_tokens) == 0:
        #     print("No key tokens!")
        #     return {"longppl": None, "n_key_token": None, "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}

        # loss_key = loss_overall[key_tokens]

        # return {"longppl": np.exp(loss_key.mean()), "n_key_token": len(key_tokens), "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}