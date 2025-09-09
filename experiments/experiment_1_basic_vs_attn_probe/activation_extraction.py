'''
For getting activations out from a dataset of prompt, target_idx pairs
'''
# %%
import torch
from nnsight import NNsight, LanguageModel

from torch.utils.data import Dataset, DataLoader
from einops import rearrange, reduce, einsum

import polars as pl

import torch.nn as nn

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torch.nn.functional import one_hot

from dataclasses import dataclass

from typing import List, Tuple, Optional

from enum import Enum

from prompts_dataset import DataConfig, PromptDataset, get_prompt_loader
# %%
class FinalTokenActivationsDataset(Dataset):
    '''
    stores activations (batch_size n_layers d_model) and correct_idx (batch_size)
    '''
    def __init__(self, final_token_activations_pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        '''
        takes in pair of (activations_tensor, correct_idx_tensor)
        '''
        self.activations = torch.concat([x[0] for x in final_token_activations_pairs]) # b l m

        self.correct_idx = torch.concat([x[1] for x in final_token_activations_pairs])
    
    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.correct_idx[idx]


'''
We expect the X to be a list of prompts, and y to be target index
'''
class ActivationExtractor:
    def __init__(self, data_config: DataConfig, prompt_loader: Optional[DataLoader] = None):
        self.data_config = data_config
        if prompt_loader is None:
            self.prompt_loader = get_prompt_loader(data_config)
        else:
            self.prompt_loader = prompt_loader

        #load the llm from huggingface via nnisght wrapper
        self.llm = LanguageModel(self.data_config.llm_name, device_map = "auto", dispatch=True)

    def get_final_token_activations_dataset(self):
        final_token_activations_pairs=[]
        self.llm.eval()
        for X, y in self.prompt_loader:

            X = [self.llm.tokenizer.apply_chat_template([[
                {"role": "user", "content": input_str}
            ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]

            seq_idxs = [len(self.llm.tokenizer.tokenize(input_str)) - 1 for input_str in X]
            with torch.no_grad():
                with self.llm.trace(X) as tracer:
                    activations= rearrange(
                        torch.stack([layer.output[range(len(seq_idxs)),seq_idxs,:] for layer in self.llm.model.layers]),#llm.transformer.h]),
                        'l b h -> b l h'
                    )
                    final_token_activations_pairs.append((activations,y))

        #dataset = (torch.concat([x[0] for x in final_token_activations]), torch.concat([x[1] for x in final_token_activations]))

        final_token_activations_dataset = FinalTokenActivationsDataset(final_token_activations_pairs)
        return final_token_activations_dataset