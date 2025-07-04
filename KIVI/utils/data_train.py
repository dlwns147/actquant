import numpy as np
import torch
import random
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader

class TextDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, tokenizer, seqlen, col_key, cutoff=1000):
        self.tokenizer = tokenizer
        self.col_key = col_key
        self.cutoff = cutoff
        self.block_size = seqlen
        if cutoff is None:
            cutoff = len(data)
        tokenized_datasets = [self.tokenizer(data[i][col_key]) for i in range(cutoff)]
        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    

# class TextDataset(Dataset):
#     def __init__(self, data, tokenizer, seqlen, col_key, per_device_train_batch_size, mode="train", cutoff=None):
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.col_key = col_key
#         self.cutoff = cutoff
#         self.seqlen = seqlen
#         self.per_device_train_batch_size = per_device_train_batch_size

#         if self.mode == "train":
#             self.encodings = self.process_data(data)
#         else:
#             self.encodings = self.process_data(data, is_val=True)

#     def process_data(self, data, is_val=False):
#         seqlen = self.seqlen
#         if is_val:
#             if self.cutoff is None:
#                 enc = self.tokenizer(" ".join(data[self.col_key]), return_tensors='pt')
#             else:
#                 enc = self.tokenizer(" ".join(data[:self.cutoff][self.col_key]), return_tensors='pt')
#             tot_num_seq = enc['input_ids'].size(1) // seqlen
#             enc['input_ids'] = enc['input_ids'][..., :tot_num_seq*seqlen]
#         else:
#             if self.cutoff is None:
#                 enc = self.tokenizer(" ".join(data[self.col_key]), return_tensors='pt')
#             else:
#                 enc = self.tokenizer(" ".join(data[:self.cutoff][self.col_key]), return_tensors='pt')
#             tot_num_seq = enc['input_ids'].size(1) // (seqlen*self.per_device_train_batch_size)
#             enc['input_ids'] = enc['input_ids'][..., :tot_num_seq*seqlen*self.per_device_train_batch_size]
        
#         return enc

#     def __getitem__(self, idx):
#         input_ids = self.encodings['input_ids'][0, idx*self.seqlen:(idx+1)*self.seqlen]
#         return input_ids

#     def __len__(self):
#         return self.encodings['input_ids'].size(1) // self.seqlen
    

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def get_c4(n_train_samples, n_eval_samples, seqlen, tokenizer):
    # raw_tra_data = load_dataset("c4", split="train")
    raw_tra_data = load_dataset('allenai/c4', 
                                data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
                                split='train')
    # raw_val_data = load_dataset("c4", split="validation")
    raw_val_data = load_dataset('allenai/c4', 
                                data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                                split='validation')
    train_dataset = TextDataset(raw_tra_data, tokenizer, 
                                col_key='text', 
                                cutoff=n_train_samples, 
                                seqlen=seqlen)
    val_dataset = TextDataset(raw_val_data, tokenizer,
                              col_key='text', 
                              cutoff=n_eval_samples, # todo: change to 1100
                              seqlen=seqlen)
    return train_dataset, val_dataset


def get_loaders(
    name, enc, n_train_samples=128, n_eval_samples=1024, seqlen=2048):
    if 'c4' in name:
        return get_c4(n_train_samples, n_eval_samples, seqlen, enc)
    else:
        raise NotImplementedError