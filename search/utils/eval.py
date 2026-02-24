import time
from functools import partial
from statistics import median
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
from .data import *
from .loss import JSD, TopK
from .func import clean_up


def get_loss_mask(seq_shift_labels, key_tokens=None, last_tokens=None, ignore_index=-100, device=None):
    """
    Create a boolean mask for loss/logits based on labels, key_token, and last_tokens.
    Used in eval_loss and get_logits to keep only positions that participate in loss.

    Parameters:
        seq_shift_labels: [seq_len-1] tensor of labels (shifted, i.e. labels[:, 1:])
        key_tokens: Optional list of token indices to keep. If None, all valid positions are kept.
        last_tokens: Optional int. If set, keep only the last N positions.
        ignore_index: Label value to ignore (excluded from mask).
        device: Device for new tensors; defaults to seq_shift_labels.device.

    Returns:
        mask: [seq_len-1] bool tensor, True where loss is computed.
    """
    if device is None:
        device = seq_shift_labels.device
    mask = (seq_shift_labels != ignore_index)
    if key_tokens is not None:
        key_mask = torch.zeros_like(mask, dtype=torch.bool)
        key_mask[key_tokens] = True
        mask = mask & key_mask
    if last_tokens is not None:
        seq_len = seq_shift_labels.size(0)
        start_idx = max(0, seq_len - last_tokens)
        last_tokens_mask = torch.arange(seq_len, device=device) >= start_idx
        mask = mask & last_tokens_mask
    return mask


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, model_name='', device=torch.device("cuda:0"), dataset='wikitext2', seqlen=2048, testloader=None, tokenizer=None):
    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model_name)

        testloader = get_loader(name=dataset, train=False, seed=0, seqlen=seqlen, tokenizer=tokenizer)
        print(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, seqlen=seqlen, device=device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, accelerator, loader, seqlen=2048):
    # # Get input IDs
    # testenc = testenc.input_ids

    # # Calculate number of samples
    # n_sample = testenc.numel() // seqlen

    # List to store negative log likelihoods
    nlls = []
    # print(f"n_sample {n_sample}")
    
    # Loop through each batch
    # for inputs in tqdm(loader, desc='Eval PPL'):
    for inputs, attention_mask, labels in tqdm(loader, desc='Eval PPL'):

        # Forward pass through the model
        # outputs = model(inputs)
        outputs = model(inputs, attention_mask=attention_mask)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * lm_logits.shape[0]

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # # Loop through each batch
    # for i in tqdm(range(0,n_sample,bs), desc='Eval PPL'):

    #     # Calculate end index
    #     j = min(i+bs, n_sample)

    #     # Prepare inputs and move to device
    #     inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
    #     inputs = inputs.reshape(j-i, seqlen)

    #     # Forward pass through the model
    #     lm_logits = model(inputs).logits

    #     # Shift logits and labels for next token prediction
    #     shift_logits = lm_logits[:, :-1, :].contiguous()
    #     shift_labels = inputs[:, 1:]

    #     # Compute loss
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    #     # Calculate negative log likelihood
    #     neg_log_likelihood = loss.float() * seqlen * (j-i)

    #     # Append to list of negative log likelihoods
    #     nlls.append(neg_log_likelihood)

    # print(f'{accelerator.device} nlls : {len(nlls)}')
    # nlls = accelerator.gather_for_metrics(nlls)
    # print(f'{accelerator.device} gathered nlls : {len(nlls)}')
    # nlls = torch.cat(nlls)
    # print(f'{accelerator.device} torch nlls : {nlls.shape}')
    nlls = torch.stack(accelerator.gather_for_metrics(nlls)).flatten()

    # Compute perplexity
    # ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    ppl = torch.exp(nlls.sum() / (len(nlls) * seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()

@torch.no_grad()
def get_logits(model, loader, key_token_list=None, last_tokens=None, ignore_index=-100):
    """
    Get model logits for each batch, storing only positions that participate in loss
    (masked by get_loss_mask) to reduce memory.

    Parameters:
        model: The model to run.
        loader: DataLoader yielding (inputs, attention_mask, labels).
        key_token_list: Optional list of key token indices per batch/seq (same format as in eval_loss).
        last_tokens: Optional int; only last N positions per sequence are kept.
        ignore_index: Label value to ignore.

    Returns:
        dense_logits_list: List of batches. Each batch is a list of tensors of shape
            [num_masked, vocab_size] per sequence (masked positions only).
    """
    if key_token_list is not None:
        assert len(loader) == len(key_token_list)
    dense_logits_list = []
    for batch_idx, (inputs, attention_mask, labels) in enumerate(loader):
        outputs = model(inputs)
        lm_logits = outputs.logits[:, :-1, :].contiguous().detach()
        shift_labels = labels[:, 1:].contiguous()
        batch_size = lm_logits.shape[0]
        batch_masked = []
        for seq_idx in range(batch_size):
            seq_shift_labels = shift_labels[seq_idx]
            key_tokens = key_token_list[batch_idx][seq_idx] if key_token_list is not None else None
            if key_tokens is None and key_token_list is not None:
                batch_masked.append(torch.empty(0, lm_logits.size(-1), device=lm_logits.device, dtype=lm_logits.dtype))
                continue
            mask = get_loss_mask(seq_shift_labels, key_tokens=key_tokens, last_tokens=last_tokens, ignore_index=ignore_index, device=lm_logits.device)
            logits_s = lm_logits[seq_idx][mask]
            batch_masked.append(logits_s)
        del lm_logits
        clean_up()

        dense_logits_list.append(batch_masked)
    return dense_logits_list


@torch.no_grad()
def eval_loss(model, accelerator, loader, seqlen=2048, loss_func='cross_entropy', dense_logits_list=None, key_token_list=None, stride=None, last_tokens=None, ignore_index=-100):
    """
    Evaluate loss on a model using a data loader.
    
    Parameters:
        model: The model to evaluate
        accelerator: Accelerator object for distributed training
        loader: DataLoader containing input data
        seqlen: Sequence length (for compatibility)
        loss_func: 'cross_entropy' or 'jsd'
        dense_logits_list: List of dense logits for JSD calculation (required for 'jsd')
        key_token_list: Pre-computed key token list (format: [batch_idx][seq_idx] -> list of token indices)
                       If provided, only key tokens are used for loss calculation.
                       If None, all tokens (except ignore_index) are used.
        ignore_index: Index to ignore in loss calculation
        k: Top-k parameter (for topk loss, not used in cross_entropy/jsd)
    
    Returns:
        Average loss value
    """
    if loss_func == 'jsd':
        assert dense_logits_list is not None, "dense_logits_list must be provided for jsd"
    if key_token_list is not None:
        assert len(loader) == len(key_token_list)
  
    # List to store losses and sequence lengths
    losses = []
    seqlens = []
    
    # Loop through each batch
    for batch_idx, (inputs, attention_mask, labels) in enumerate(loader):
        batch_size = inputs.shape[0]
        
        # Forward pass
        outputs = model(inputs, attention_mask=attention_mask)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Process each sequence in the batch
        batch_losses = []
        batch_seqlens = []
        
        for seq_idx in range(batch_size):
            # Get sequence-specific logits and labels
            seq_shift_logits = shift_logits[seq_idx]  # [seq_len-1, vocab_size]
            seq_shift_labels = shift_labels[seq_idx]  # [seq_len-1]

            key_tokens = key_token_list[batch_idx][seq_idx] if key_token_list is not None else None
            if key_tokens is None and key_token_list is not None:
                continue
            mask = get_loss_mask(seq_shift_labels, key_tokens=key_tokens, last_tokens=last_tokens, ignore_index=ignore_index, device=seq_shift_labels.device)

            cur_seqlen = mask.sum().item()
            if cur_seqlen == 0:
                continue
            
            # Reshape for loss computation
            seq_shift_logits = seq_shift_logits.reshape(-1, seq_shift_logits.size(-1))
            seq_shift_labels = seq_shift_labels.reshape(-1)
            
            # Compute loss on selected tokens
            if loss_func == 'cross_entropy':
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
                loss = loss_fct(seq_shift_logits[mask], seq_shift_labels[mask])
            
            elif loss_func == 'jsd':
                # Dense logits for this sequence are already masked (same mask as seq_shift_logits)
                dense_logits_seq = dense_logits_list[batch_idx][seq_idx].contiguous()
                # Compute JSD on selected tokens
                loss_fct = JSD()
                loss = loss_fct(seq_shift_logits[mask], dense_logits_seq)
            
            else:
                raise NotImplementedError(f'{loss_func} is not implemented')
            
            # Weight loss by sequence length
            loss = loss.float() * cur_seqlen
            batch_losses.append(loss)
            batch_seqlens.append(cur_seqlen)
        
        # Aggregate batch losses
        if len(batch_losses) > 0:
            losses.extend(batch_losses)
            seqlens.extend(batch_seqlens)

    # Compute average loss
    if len(losses) == 0:
        return 0.0
    
    losses = torch.stack(accelerator.gather_for_metrics(losses)).flatten()
    total_seqlen = sum(accelerator.gather_for_metrics(seqlens))
    
    if total_seqlen > 0:
        loss_sum = losses.sum() / total_seqlen
    else:
        loss_sum = torch.tensor(0.0)

    return loss_sum.item()


def eval_metric(model, accelerator, metric, loader, seqlen, loss_func='cross_entropy', dense_logits_list=None, key_token_list=None, stride=None, last_tokens=None, tokenizer=None, limit=None, batch_size=None, num_fewshot=None, verbosity='INFO', task_manager=None, task_dict=None):
    """
    Evaluate metric on a model using a data loader.
    
    Supported metrics:
        - 'ppl': Perplexity
        - 'loss': Loss (cross_entropy or jsd)
        - 'gsm8k': GSM8K zero-shot evaluation
    
    Parameters:
        model: The model to evaluate
        accelerator: Accelerator object for distributed training
        metric: Metric to evaluate ('ppl', 'loss', or 'gsm8k')
        loader: DataLoader containing input data
        seqlen: Sequence length
        loss_func: 'cross_entropy' or 'jsd' (for 'loss' metric)
        dense_logits_list: List of dense logits for JSD calculation (required for 'jsd')
        key_token_list: Pre-computed key token list (optional, for 'loss' metric)
        stride: Stride for stride-aware loss calculation
        last_tokens: If set, loss is computed only on the last N tokens per sequence (for 'loss' metric)
        tokenizer: Tokenizer (required for 'gsm8k')
        limit: Limit number of samples (for 'gsm8k')
        batch_size: Batch size (for 'gsm8k')
        num_fewshot: Number of few-shot examples (for 'gsm8k')
        verbosity: Verbosity level (for 'gsm8k')
        task_manager: Task manager (for 'gsm8k')
        task_dict: Task dictionary (for 'gsm8k')
    
    Returns:
        Metric value
    """
    if metric == 'ppl':
        return eval_ppl(model, accelerator, loader, seqlen=seqlen)
    elif metric == 'loss':
        return eval_loss(model, accelerator, loader, seqlen=seqlen, loss_func=loss_func, dense_logits_list=dense_logits_list, key_token_list=key_token_list, stride=stride, last_tokens=last_tokens)
    elif 'gsm8k' in metric:
        return eval_zeroshot(model, tokenizer, task_list=[metric], limit=limit, batch_size=batch_size, num_fewshot=num_fewshot, verbosity=verbosity, task_manager=task_manager, task_dict=task_dict)
    else:
        raise NotImplementedError(f'{metric} is not supported')


@torch.no_grad()
def measure_latency(model, generation, device, batch_size=64, prompt_length=64, generation_length=128, iteration=10, max_time=1e9) :

    def cuda_timestamp(sync=False, device=None):
        if sync:
            torch.cuda.synchronize(device=device)
        return time.perf_counter()

    time_fn = partial(cuda_timestamp, device=device)

    def _step(input):
        t_step_start = time_fn()
        model(input)
        t_step_end = time_fn(True)
        return t_step_end - t_step_start

    def _step_gen(input, generation_length):
        t_step_start = time_fn()
        model.generate(input,min_new_tokens=generation_length, max_new_tokens=generation_length)
        t_step_end = time_fn(sync=True)
        return t_step_end - t_step_start
    
    latency = []
    if (generation) :
        # setting for token generation
        max_length = prompt_length + generation_length
        model.config.max_length = max_length
        config_use_cache = model.config.use_cache
        generation_config_use_cache = model.generation_config.use_cache
        model.config.use_cache = True
        model.generation_config.use_cache = True

        # make dummy input
        random_input = torch.randint(0, 31999, (batch_size, prompt_length), dtype=torch.long)
        random_input = random_input.to(device).contiguous()

        # dummy inference
        model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)

        # latency for 10 iterations
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in tqdm(range(iteration)):
            # starter.record()
            # model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)
            # ender.record()
            # torch.cuda.synchronize()
            # cur_time = starter.elapsed_time(ender)
            cur_time = max_time
            try:
                cur_time = _step_gen(random_input, generation_length)
            except RuntimeError:
                pass
            latency.append(cur_time)

    else :
        # setting for prompt processing
        # batch_size = 1
        config_use_cache = model.config.use_cache
        generation_config_use_cache = model.generation_config.use_cache
        model.config.use_cache = False
        model.generation_config.use_cache = False
        # iteration = 50

        # make dummy input for module.weight shape
        random_input = torch.randint(0, 31999, (batch_size, 2048), dtype=torch.long)
        random_input = random_input.to(device).contiguous()
        
        # dummy inference
        model(random_input)

        # latency for 50 iterations
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in tqdm(range(iteration)):
            # starter.record()
            # model(random_input)
            # ender.record()
            # torch.cuda.synchronize()
            # cur_time = starter.elapsed_time(ender)
            cur_time = _step(random_input)
            latency.append(cur_time)

    # curr_time = starter.elapsed_time(ender)
    median_latency = median(latency)
    # mean_latency = curr_time/iteration

    model.config.use_cache = config_use_cache
    model.generation_config.use_cache = generation_config_use_cache
    
    gc.collect()
    torch.cuda.empty_cache()

    return median_latency

@torch.no_grad()
def eval_zeroshot(model, tokenizer, task_list=['coqa', 'gsm8k', 'truthfulqa'], batch_size=None, task_manager=None, task_dict=None, num_fewshot=None, limit=None, verbosity='INFO'):
    
    from lm_eval.models.huggingface import HFLM
    from lm_eval import tasks, evaluator, utils
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    
    # model.tie_weights = lambda: None
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size if batch_size is not None else 1) #, batch_size='auto')
    
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        verbosity=verbosity,
        task_manager=task_manager,
        task_dict=task_dict
    )
    
    return results['results']


# @torch.no_grad()
# def eval_loss(model, accelerator, loader, seqlen=2048, loss_func='cross_entropy', dense_logits_list=None):
#     # Get input IDs
#     # testenc = testenc.input_ids

#     # Calculate number of samples
#     # n_sample = testenc.numel() // seqlen
  
#     # List to store negative log likelihoods
#     losses = []
    
#     # Loop through each batch
#     # for i, inputs in enumerate(loader):
#     for i, (inputs, attention_mask, labels) in enumerate(loader):
#         # outputs = model(inputs)
#         outputs = model(inputs, attention_mask=attention_mask)
#         lm_logits = outputs.logits

#         # Shift logits and labels for next token prediction
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        
#         # Compute loss
#         if loss_func == 'cross_entropy':
#             loss_fct = nn.CrossEntropyLoss()
#             shift_labels = inputs[:, 1:].reshape(-1)
#             loss = loss_fct(shift_logits, shift_labels)
#         elif loss_func == 'jsd':
#             assert dense_logits_list != None
#             dense_logits = dense_logits_list[i]
#             shift_dense_logits = dense_logits[:, :-1, :].reshape(-1, shift_logits.size(-1)).contiguous()
#             shift_labels = labels[:, 1:].reshape(-1, 1)
#             loss_fct = JSD()
#             loss = loss_fct(shift_logits, shift_dense_logits, label=shift_labels)
#         else:
#             raise NotImplementedError(f'{loss_func} is not implemented')

#         # Calculate negative log likelihood
#         loss = loss.float() * seqlen * lm_logits.shape[0]

#         # Append to list of negative log likelihoods
#         losses.append(loss)

#     # for i in range(0,n_sample,bs):

#     #     # Calculate end index
#     #     j = min(i+bs, n_sample)

#     #     # Prepare inputs and move to device
#     #     inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
#     #     inputs = inputs.reshape(j-i, seqlen)

#     #     # Forward pass through the model
#     #     outputs = model(inputs)
#     #     lm_logits = outputs.logits

#     #     # Shift logits and labels for next token prediction
#     #     shift_logits = lm_logits[:, :-1, :]
#     #     shift_logits = shift_logits.reshape(-1, shift_logits.size(-1)).contiguous()
#     #     shift_labels = inputs[:, 1:]

#     #     # Compute loss
#     #     if loss_func == 'cross_entropy':
#     #         loss_fct = nn.CrossEntropyLoss()
#     #         loss = loss_fct(shift_logits, shift_labels.reshape(-1))
#     #     elif loss_func == 'jsd':
#     #         dense_logits = dense_logits_list[i: j]
#     #         dense_logits = dense_logits[:, :-1, :].reshape(-1, dense_logits.size(-1)).contiguous()
#     #         loss_fct = JSD()
#     #         loss = loss_fct(shift_logits, dense_logits)
#     #     else:
#     #         raise NotImplementedError(f'{loss_func} is not implemented')

#         # # Calculate negative log likelihood
#         # loss = loss.float() * seqlen * (j-i)
#         # loss = accelerator.gather_for_metrics(loss)

#         # # Append to list of negative log likelihoods
#         # losses.append(loss)
    
#     # Compute sum of negative log_likelihood
#     # losses = accelerator.gather_for_metrics(losses)
#     # print(f'losses: {losses}, {len(losses)}')
#     # losses = torch.cat(losses)
#     losses = torch.stack(accelerator.gather_for_metrics(losses)).flatten()
#     loss_sum = losses.sum() / (len(losses) * seqlen)
#     # loss_sum = torch.stack(losses).sum() / seqlen

#     return loss_sum.item()
