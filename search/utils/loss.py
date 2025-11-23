import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

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

def get_key_token_list(model, tokenizer, loader, save_path=None, trunc_len=4096, sliding_window=1024, alpha=2, beta=-2):
    text_list = []
    for input_ids_batch, _, _ in loader:
        for input_ids in input_ids_batch:
            text_list.append(tokenizer.decode(input_ids))
    
    key_token_list = []
    for text in text_list:        
        encoded_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        input_ids = encoded_input['input_ids'].to(model.device)
        offset_mapping = encoded_input['offset_mapping'].reshape(-1, 2)
    
        key_text_slices = find_key_token(text, model, tokenizer, trunc_len, sliding_window, save_path, alpha, beta)
        key_tokens = cal_overlap(offset_mapping, key_text_slices)
        key_token_list.append(key_tokens)
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

def find_key_token(text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path=None, alpha=2, beta=-2):
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
            shift_full_trunc_logits = shift_full_logits[:, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :].reshape(-1, shift_full_logits.size(-1))
            
            shift_full_labels = input_ids[:, start_token+trunc_len: start_token+trunc_len+sliding_window].reshape(-1)
            shift_short_labels = input_ids_short[:, trunc_len: trunc_len+sliding_window].reshape(-1)

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
    key_text_intervals = merge_intervals(text_encoded['offset_mapping'].reshape(-1, 2)[key_tokens])

    if save_path is not None:
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
    total_offsets = []
    for input_id in input_ids:
        offsets = []
        text_pointer = 0
        for i, input_id in enumerate(input_id):
            if i == 0:
                token_len = len(tokenizer.decode(input_ids[0, 0]))
            else:
                token_len = len(tokenizer.decode(input_ids[0, i-1:i+1])) - len(tokenizer.decode(input_ids[0, i-1:i]))
            
            offsets.append([text_pointer, text_pointer+token_len])
            text_pointer += token_len
        total_offsets.append(offsets)

    return torch.tensor(offsets)


def compute_longppl(
        text,
        model,
        evaluator_model,
        tokenizer=None, 
        evaluator_tokenizer=None, 
        save_path=None, 
        trunc_len=4096, 
        sliding_window=1024,
        alpha=2,
        beta=-2
    ):
    r"""
    Compute the LongPPL for long text sequences.

    Parameters:
        text (`str`): 
            The input text for which LongPPL is calculated.
        model (`transformers.PretrainedModel` or `str`): 
            Can be either:
                - The model used for LongPPL calculation.
                - The path to the model used for LongPPL calculation.
        evaluator_model (`transformers.PretrainedModel` or `str`): 
            Can be either:
                - The evaluator model used to identify the key tokens.
                - The path to the evaluator model.
        tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
            Tokenizer of the evaluated model if `model` is specified with a `transformers.PretrainedModel` object, otherwise should be `None`.
        evaluator_tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
            Tokenizer of the evaluator model if `evaluator_model` is specified with a `transformers.PretrainedModel` object, otherwise should be `None`.
        save_path (`str`, *optional*): If specified, the path to save the computed key tokens.
        trunc_len (`int`, *optional*, default=4096): Length of the truncated short context.
        sliding_window (`int`, *optional*, default=1024): Number of tokens sharing the same short context.

    Returns:
        [`Dict[np.float32, int, np.float32, int]`]: A `Dict` object including:
            - 'longppl' (`np.float32`): The LongPPL score.
            - 'n_key_token' (`int`): The number of key tokens (under the evaluated model).
            - 'ppl' (`np.float32`): The PPL score.
            - 'n_token' (`int`): The number of tokens in the input text.
    """
    assert type(text) in [str, list]
    if type(text) == str:
        text = [text]
    total_seqlen = 0
    total_key_token_len = 0
    nll_all_list = []
    nll_key_list = []
    for cur_text in text:
        try:
            encoded_input = tokenizer(cur_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            input_ids = encoded_input['input_ids'].to(model.device)
            offset_mapping = encoded_input['offset_mapping'][0]
        except NotImplementedError:
            encoded_input = tokenizer(cur_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False)
            input_ids = encoded_input['input_ids'].to(model.device)
            offset_mapping = compute_offsets(cur_text, tokenizer, input_ids)[0]
            
        if evaluator_model is not None:
            key_text_slices = find_key_token(cur_text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path, alpha, beta)
        else:
            key_text_slices = load_key_token(save_path)

        key_tokens = cal_overlap(offset_mapping, key_text_slices)
        bs, seqlen = input_ids.shape
        key_token_len = len(key_tokens)
        
        with torch.no_grad():
            outputs = model(input_ids)
        lm_logits = outputs.logits
            
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = input_ids[:, 1:].reshape(-1)
        
        loss_func = torch.nn.CrossEntropyLoss()
        loss_all = loss_func(shift_logits, shift_labels)
        loss_key = loss_func(shift_logits[key_tokens], shift_labels[key_tokens])        
        
        nll_all = loss_all.float() * seqlen * bs
        nll_all_list.append(nll_all)
        total_seqlen += seqlen * bs
        
        if key_tokens is not None and len(key_tokens) > 0:
            nll_key = loss_key.float() * key_token_len * bs
            nll_key_list.append(nll_key)
            total_key_token_len += key_token_len * bs
        
        decode_key_tokens = tokenizer.decode(shift_labels[key_tokens])
        print(f'decode_key_tokens: {decode_key_tokens}')

        # loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        # loss_all = loss_func(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float)
        # loss_key = loss_func(output_full.logits[0, :-1][key_tokens], input_ids[0, 1:][key_tokens])
        
    # if key_tokens is None or len(key_tokens) == 0:
    #     print("No key tokens!")
    #     return {"longppl": None, "n_key_token": None, "ppl": torch.exp(loss_all), "n_token": input_ids.shape[-1]}
    # return {"longppl": torch.exp(loss_key), "n_key_token": len(key_tokens), "ppl": torch.exp(loss_all), "n_token": input_ids.shape[-1]}
    
    ppl_all = torch.exp(sum(nll_all_list) / total_seqlen)
    ppl_key = torch.exp(sum(nll_key_list) / total_key_token_len) if total_key_token_len > 0 else None
    return {"longppl": ppl_key, "n_key_token": total_key_token_len, "ppl": ppl_all, "n_token": total_seqlen}

    
        
        # loss_f = torch.nn.CrossEntropyLoss(reduction='none')
        # loss_overall = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float).cpu().numpy()
        
        # if key_tokens is None or len(key_tokens) == 0:
        #     print("No key tokens!")
        #     return {"longppl": None, "n_key_token": None, "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}

        # loss_key = loss_overall[key_tokens]

        # return {"longppl": np.exp(loss_key.mean()), "n_key_token": len(key_tokens), "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}