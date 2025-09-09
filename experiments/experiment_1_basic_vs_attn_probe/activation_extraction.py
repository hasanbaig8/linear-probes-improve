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

class AllTokenActivationsDataset(Dataset):
    def __init__(self, all_token_activation_pairs: List[Tuple[torch.Tensor, torch.Tensor]], seq_idxs: List[int]):
        self.all_token_activation_pairs = all_token_activation_pairs
        self.seq_idxs = seq_idxs
    
    def __len__(self):
        return len(self.all_token_activation_pairs)
    
    def __getitem__(self, idx: int):
        return self.all_token_activation_pairs[idx][0][:,:self.seq_idxs[idx],:], self.all_token_activation_pairs[idx][0] #first index batch, then activation, then all layers, then up to the sequence posn, then all d_model

'''
We expect the X to be a list of prompts, and y to be target index
'''
class ActivationExtractor:
    def __init__(self, data_config: DataConfig, prompt_loader: Optional[DataLoader] = None, llm: Optional[LanguageModel] = None):
        self.data_config = data_config
        if prompt_loader is None:
            self.prompt_loader = get_prompt_loader(data_config)
        else:
            self.prompt_loader = prompt_loader

        #load the llm from huggingface via nnisght wrapper
        if llm is None:
            self.llm = LanguageModel(self.data_config.llm_name, device_map = "auto", dispatch=True)
        else:
            self.llm = llm

    def get_final_token_activations_dataset(self) -> FinalTokenActivationsDataset:
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
    
    def get_all_token_activations_dataset(self):
        all_token_activation_pairs=[]
        seq_idxs_list = []
        self.llm.eval()
        for X, y in self.prompt_loader:

            X = [self.llm.tokenizer.apply_chat_template([[
                {"role": "user", "content": input_str}
            ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]

            seq_idxs = [len(self.llm.tokenizer.tokenize(input_str))  for input_str in X]
            with torch.no_grad():
                with self.llm.trace(X) as tracer:
                    activations= rearrange(
                        torch.stack([layer.output for layer in self.llm.model.layers]),
                        'l b s h -> b l s h'
                    )
                    
                    # Add each prompt individually to the dataset
                    for i in range(len(X)):
                        all_token_activation_pairs.append((activations[i:i+1], y[i:i+1]))
                        seq_idxs_list.append(seq_idxs[i])

        #dataset = (torch.concat([x[0] for x in final_token_activations]), torch.concat([x[1] for x in final_token_activations]))

        all_token_activations_dataset = AllTokenActivationsDataset(all_token_activation_pairs, seq_idxs_list)
        return all_token_activations_dataset