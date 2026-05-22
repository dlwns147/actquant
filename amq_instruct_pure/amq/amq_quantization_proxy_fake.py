import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='meta-llama')
parser.add_argument('--model_name', type=str, default='Llama-3.1-8B-Instruct')
parser.add_argument('--save_path', type=str, default='/SSD/hqq')
parser.add_argument('--nbits', type=int, default=2)
parser.add_argument('--group_size', type=int, default=128)
args = parser.parse_args()

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_id  = f'{args.model_path}/{args.model_name}'

nbits=args.nbits
group_size=args.group_size
axis=1

#Load model on the CPU
######################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
model      = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

#Quantize with HQQ, then copy the dequantized (fake-quant) weights into base_model
######################################################################################

if nbits < 16:
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    AutoHQQHFModel.quantize_model(
        model, quant_config=quant_config,
        compute_dtype=torch.float16, device='cuda',
    )

    # For every HQQLinear in the quantized tree, copy its dequantized fp16
    # weight into the matching nn.Linear in base_model. base_model stays a
    # standard HF model end-to-end (no HQQLinear anywhere), so generate()
    # runs on the normal transformers forward path with a real KV cache.
    base_linears = dict(base_model.named_modules())
    with torch.no_grad():
        for qname, qmod in model.named_modules():
            if not isinstance(qmod, HQQLinear):
                continue
            target = base_linears.get(qname)
            if target is None:
                raise KeyError(f"base_model has no module named {qname}")
            w = qmod.dequantize().to(torch.float16).detach().cpu()
            target.weight.data.copy_(w)
            qb = getattr(qmod, "bias", None)
            if qb is not None and target.bias is not None:
                target.bias.data.copy_(qb.detach().to(torch.float16).cpu())

    # free the HQQ tree; base_model is what we save
    del model
    torch.cuda.empty_cache()

save_dir = os.path.join(
    args.save_path,
    f'{args.model_name}_{args.nbits}bit_{args.group_size}gs_{axis}axis_fake',
)
os.makedirs(save_dir, exist_ok=True)
base_model.save_pretrained(save_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(model_id).save_pretrained(save_dir)
