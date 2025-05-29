import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

# import sys
# sys.path.append('..')
from .base import BASE, get_awq_calib_dataset
# from utils.dispatch import simple_dispatch_model
from accelerate import dispatch_model

from .awq_utils.pre_quant import run_awq, apply_awq

import gc
from tqdm import tqdm
import math
import functools
from copy import deepcopy

from collections import defaultdict

class AWQ(BASE):
    def __init__(self, model_name, config, arch, device_map, group_size=128, dev='cuda', prune=False, do_owq=False, owq=None, **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map, group_size=group_size, dev=dev, prune=prune, do_owq=do_owq, owq=owq)
        self.method = 'awq'

        self.clip_asym = kwargs.get('clip_asym', True)
        if self.clip_asym:
            print("Clipping asymmetrically")
        else:
            print("Clipping symmetrically")


    def run(self, nsamples=128, seqlen=512, no_zero_point=False):    
        q_config = {
            "zero_point": not no_zero_point,  # by default True
            "q_group_size": self.group_size,  # whether to use group quantization
        }
        print("Quantization config:", q_config)
        awq_results = run_awq(self.model, self.tokenizer, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym, n_samples=nsamples, seqlen=seqlen, do_owq=self.do_owq, outlier=self.owq)
        self.load_model(device_map='cpu')
        # self.model = simple_dispatch_model(self.model, self.device_map)
        self.model = dispatch_model(self.model, self.device_map)
        apply_awq(self.model, awq_results, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym, do_owq=self.do_owq, outlier=self.owq)
        torch.cuda.empty_cache()
        gc.collect()