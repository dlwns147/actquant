
import torch
from datasets import load_dataset

from utils.data import get_loader, get_tokenizer

tokenizer = get_tokenizer('/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct')
loader = get_loader('minilongbench', tokenizer=tokenizer, require_answer=True)
import pdb; pdb.set_trace()