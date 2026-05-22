import utils
import model_utils
import quant_utils
import torch
import os
import logging
from tqdm import tqdm
import torch.nn as nn
from time import time
import gc
import torch.functional as F


@torch.no_grad()
def evaluator(model, testenc, dev, dataset, args, layerwise=True):

    model.eval()

    if 'opt' in args.model:
        opt_type = True
        llama_type = False
    elif 'meta' in args.model:
        llama_type = True
        opt_type = False
    else:
        raise ValueError(f'Unknown model {args.model}')


    use_cache = model.config.use_cache
    model.config.use_cache = False

    if layerwise:
        if opt_type:
            layers = model.model.decoder.layers
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

        elif llama_type:
            layers = model.model.layers
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            if hasattr(model.model, 'rotary_emb'):
                model.model.rotary_emb.to(dev)

        layers[0] = layers[0].to(dev)

        # Convert the whole text of evaluation dataset into batches of sequences.
        input_ids = testenc.input_ids  # (1, text_len)
        nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
        input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

        batch_size = args.bsz
        input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
        nbatches = len(input_ids)

        dtype = next(iter(model.parameters())).dtype
        # The input of the first decoder layer.
        inps = torch.zeros(
            (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        inps = [0] * nbatches
        cache = {'i': 0, 'attention_mask': None}
        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                if llama_type:
                    cache['position_ids'] = kwargs['position_ids']
                raise ValueError
        layers[0] = Catcher(layers[0])
    
        for i in range(nbatches):
            batch = input_ids[i]
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()

        if opt_type:
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.cpu()
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        elif llama_type:
            position_ids = cache['position_ids']
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            if hasattr(model.model, 'rotary_emb'):
                model.model.rotary_emb.cpu()

        torch.cuda.empty_cache()
        outs = [0] * nbatches
        attention_mask = cache['attention_mask']

        for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
            # import pdb; pdb.set_trace()
            layer = layers[i].to(dev)

            # Dump the layer input and output
            if args.capture_layer_io and args.layer_idx == i:
                captured_io = model_utils.capture_layer_io(model_utils.get_model_type(model), layer, inps)
                save_path = model_utils.get_layer_io_save_path(args)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(captured_io, save_path)
                logging.info(f'Dumped layer input and output to: {save_path}')

            # torch.cuda.synchronize()
            # layer_forward_start = time()
            for j in range(nbatches):
                # torch.cuda.synchronize()
                # batch_start = time()
                if opt_type:
                    outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
                elif llama_type:
                    outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
                # torch.cuda.synchronize()
                # batch_time = time() - batch_start
                # print(f'batch_time : {batch_time}')
            # torch.cuda.synchronize()
            # layer_forward_time = time() - layer_forward_start
            # print(f'layer_forward_time : {layer_forward_time}')
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            gc.collect()
            inps, outs = outs, inps

        if opt_type:
            if model.model.decoder.final_layer_norm is not None:
                model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
            if model.model.decoder.project_out is not None:
                model.model.decoder.project_out = model.model.decoder.project_out.to(dev)

        elif llama_type:
            if model.model.norm is not None:
                model.model.norm = model.model.norm.to(dev)

        model.lm_head = model.lm_head.to(dev)
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
        for i in range(nbatches):
            hidden_states = inps[i]
            if opt_type:
                if model.model.decoder.final_layer_norm is not None:
                    hidden_states = model.model.decoder.final_layer_norm(hidden_states)
                if model.model.decoder.project_out is not None:
                    hidden_states = model.model.decoder.project_out(hidden_states)
            elif llama_type:
                if model.model.norm is not None:
                    hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :]
            shift_labels = input_ids[i][:, 1:]
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            neg_log_likelihood = loss.float().mean(dim=1)
            nlls.append(neg_log_likelihood)
        nlls_tensor = torch.cat(nlls)
        ppl = torch.exp(nlls_tensor.mean())
        model.config.use_cache = use_cache
        logging.info(f'\n{dataset} PPL: {ppl.item()}')
        return ppl.item()
    
    else:
        if args.distribute:
            utils.distribute_model(model)
        else:
            model.to(dev)

        # Get input IDs
        testenc = testenc.input_ids
        seqlen = model.seqlen

        # Calculate number of samples
        nsamples = testenc.numel() // seqlen

        # List to store negative log likelihoods
        nlls = []
        # print(f"nsamples {nsamples}")

        # Loop through each batch
        for i in tqdm(range(0,nsamples,args.bsz)):

            # Calculate end index
            j = min(i+args.bsz, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:,(i * seqlen):(j * seqlen)].to(dev)
            inputs = inputs.reshape(j-i, seqlen)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * seqlen * (j-i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        model.to('cpu')
        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        model.config.use_cache = use_cache
        logging.info(f'\n{dataset} PPL: {ppl.item()}')
        return ppl.item()


class JSD(nn.Module):
    def __init__(self, tau=1., reduction='batchmean'):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
        # self.kl = nn.KLDivLoss(reduction='sum', log_target=True)
        self.tau = tau

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = (p / self.tau).log_softmax(-1), (q / self.tau).log_softmax(-1)
        m = (0.5 * (p + q))
        return 0.5 * (self.kl(m, p) + self.kl(m, q))


@torch.no_grad()
def get_loss(model,
             testenc,
             bs=1,
             loss_func='cross_entropy',
             dense_logits_list=None,
             tau=1,
             device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
  
    # List to store negative log likelihoods
    losses = []
    
    # Loop through each batch
    for i in range(0,nsamples,bs):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :]
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1)).contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        if loss_func == 'cross_entropy':
            loss = F.cross_entropy(shift_logits, shift_labels.reshape(-1))
            # loss = nn.CrossEntropyLoss()(shift_logits, shift_labels.reshape(-1))

            # Calculate negative log likelihood
            loss = loss.float() * model.seqlen * (j-i)

        elif loss_func == 'jsd' :
            assert dense_logits_list is not None
            dense_logits = dense_logits_list[i: j]
            dense_logits = dense_logits[:, :-1, :].reshape(-1, dense_logits.size(-1)).contiguous()

            loss = JSD(tau=tau, reduction='batchmean')(shift_logits, dense_logits) * model.seqlen * (j-i)
            
        else:
            raise NotImplementedError
        # Append to list of negative log likelihoods
        losses.append(loss)
    
    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum() / (nsamples * model.seqlen)

    return loss_sum.item()


@torch.no_grad()
def get_dense_logits(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
  
    # List to store negative log likelihoods
    logits = []

    # Loop through each batch
    for i in range(0,nsamples,bs):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits
        logits.append(lm_logits)

    dense_logits_list = torch.cat(logits, dim=0)

    return dense_logits_list