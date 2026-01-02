"""
Wrapper module that extends eval_metric to support LongPPL metrics.
This file provides integration between the search module and LongPPL without modifying the original eval.py
"""

from .eval import eval_metric, eval_ppl, eval_loss
from .eval_longppl import eval_longppl, eval_metric_longppl


def eval_metric_extended(
    model, 
    accelerator, 
    metric, 
    loader, 
    seqlen, 
    loss_func='cross_entropy', 
    dense_logits_list=None, 
    key_token_list=None, 
    tokenizer=None, 
    limit=None, 
    batch_size=None, 
    num_fewshot=None, 
    verbosity='INFO', 
    task_manager=None, 
    task_dict=None,
    # LongPPL specific parameters
    evaluator_model=None,
    evaluator_tokenizer=None,
    trunc_len=4096,
    sliding_window=1024,
    alpha=2.0,
    beta=-2.0,
    save_path=None,
    mode='offline',
    evaluator_name="Meta-Llama-3.1-8B"
):
    """
    Extended version of eval_metric that supports LongPPL metrics.
    
    This function extends the original eval_metric to support:
    - 'longppl': Standard LongPPL metric
    - 'longppl_jsd': LongPPL with JSD (Jensen-Shannon Divergence) loss
    
    All original metrics ('ppl', 'loss', 'gsm8k', etc.) are still supported.
    
    Additional parameters for LongPPL:
        evaluator_model: Model used to identify key tokens (for online mode)
        evaluator_tokenizer: Tokenizer for evaluator model
        trunc_len: Length of truncated short context for LongPPL
        sliding_window: Size of sliding window for LongPPL
        alpha: Threshold for LSD in key token detection
        beta: Threshold for LCL in key token detection
        save_path: Path to save key tokens (optional)
        mode: 'online' to compute key tokens, 'offline' to use precomputed
        evaluator_name: Name of evaluator model for saving key tokens
    """
    # Handle LongPPL metrics
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
        # Fall back to original eval_metric for other metrics
        return eval_metric(
            model=model,
            accelerator=accelerator,
            metric=metric,
            loader=loader,
            seqlen=seqlen,
            loss_func=loss_func,
            dense_logits_list=dense_logits_list,
            key_token_list=key_token_list,
            tokenizer=tokenizer,
            limit=limit,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            verbosity=verbosity,
            task_manager=task_manager,
            task_dict=task_dict
        )

