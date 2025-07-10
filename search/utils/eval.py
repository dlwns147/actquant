import time
from functools import partial
from statistics import median
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
from .data import *
from .loss import JSD


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
    for inputs in tqdm(loader, desc='Eval PPL'):

        # Forward pass through the model
        outputs = model(inputs)
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
def get_logits(model, loader):    
    # List to store negative log likelihoods
    logits = []
    for inputs in loader:

        outputs = model(inputs)
        lm_logits = outputs.logits
        logits.append(lm_logits)

    dense_logits_list = torch.cat(logits, dim=0).detach()

    return dense_logits_list


@torch.no_grad()
def eval_loss(model, accelerator, loader, seqlen=2048, loss_func='cross_entropy', dense_logits_list=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # n_sample = testenc.numel() // seqlen
  
    # List to store negative log likelihoods
    losses = []
    
    # Loop through each batch
    for i, inputs in enumerate(loader):

        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)
        
        # Compute loss
        if loss_func == 'cross_entropy':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
        elif loss_func == 'jsd':
            dense_logits = dense_logits_list[i]
            dense_logits = dense_logits[:-1, :].reshape(-1, shift_logits.size(-1)).contiguous()
            loss_fct = JSD()
            loss = loss_fct(shift_logits, dense_logits)
        else:
            raise NotImplementedError(f'{loss_func} is not implemented')

        # Calculate negative log likelihood
        loss = loss.float() * seqlen * lm_logits.shape[0]

        # Append to list of negative log likelihoods
        losses.append(loss)

    # for i in range(0,n_sample,bs):

    #     # Calculate end index
    #     j = min(i+bs, n_sample)

    #     # Prepare inputs and move to device
    #     inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
    #     inputs = inputs.reshape(j-i, seqlen)

    #     # Forward pass through the model
    #     outputs = model(inputs)
    #     lm_logits = outputs.logits

    #     # Shift logits and labels for next token prediction
    #     shift_logits = lm_logits[:, :-1, :]
    #     shift_logits = shift_logits.reshape(-1, shift_logits.size(-1)).contiguous()
    #     shift_labels = inputs[:, 1:]

    #     # Compute loss
    #     if loss_func == 'cross_entropy':
    #         loss_fct = nn.CrossEntropyLoss()
    #         loss = loss_fct(shift_logits, shift_labels.reshape(-1))
    #     elif loss_func == 'jsd':
    #         dense_logits = dense_logits_list[i: j]
    #         dense_logits = dense_logits[:, :-1, :].reshape(-1, dense_logits.size(-1)).contiguous()
    #         loss_fct = JSD()
    #         loss = loss_fct(shift_logits, dense_logits)
    #     else:
    #         raise NotImplementedError(f'{loss_func} is not implemented')

        # # Calculate negative log likelihood
        # loss = loss.float() * seqlen * (j-i)
        # loss = accelerator.gather_for_metrics(loss)

        # # Append to list of negative log likelihoods
        # losses.append(loss)
    
    # Compute sum of negative log_likelihood
    # losses = accelerator.gather_for_metrics(losses)
    # print(f'losses: {losses}, {len(losses)}')
    # losses = torch.cat(losses)
    losses = torch.stack(accelerator.gather_for_metrics(losses)).flatten()
    loss_sum = losses.sum() / (len(losses) * seqlen)
    # loss_sum = torch.stack(losses).sum() / seqlen

    return loss_sum.item()


def eval_metric(model, accelerator, metric, loader, seqlen, loss_func='cross_entropy', dense_logits_list=None, tokenizer=None, num_fewshot=None, limit=None, batch_size=None, verbosity='INFO'):
    # accelerator.wait_for_everyone()
    if metric == 'ppl':
        return eval_ppl(model, accelerator, loader, seqlen=seqlen)
    elif metric == 'loss':
        return eval_loss(model, accelerator, loader, seqlen=seqlen, loss_func=loss_func, dense_logits_list=dense_logits_list)
    elif 'gsm8k' in metric:
        return eval_zeroshot(model, tokenizer, task_list=[metric], num_fewshot=num_fewshot, limit=limit, batch_size=batch_size, verbosity=verbosity)
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
def eval_zeroshot(model, tokenizer, task_list=['coqa', 'gsm8k', 'truthfulqa'], batch_size=None, num_fewshot=None, limit=None, verbosity='INFO'):
    
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
        verbosity=verbosity
    )
    
    return results['results']