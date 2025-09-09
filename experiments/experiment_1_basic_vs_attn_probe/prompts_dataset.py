# %% Imports
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

# %%
@dataclass
class DataConfig:
    train_prompts_file_path: str
    chosen_subset_path: str
    n_samples_train: int
    batch_size: int
    llm_name: str

class PromptDataset(Dataset):
    def __init__(self, all_prompts_df: pl.DataFrame, chosen_path: str, n_samples: int):
        self.prompts_df = all_prompts_df.filter(pl.col('path').eq(chosen_path)).head(n_samples)
    
    def __len__(self):
        return len(self.prompts_df)

    def __getitem__(self, idx: int):
        row =self.prompts_df[idx]
        X = row['prompt'].item()
        y = row['shuffled_correct_choice_idx'].item()
        return (X,y)

def get_prompt_loader(data_config: DataConfig) -> DataLoader:
    all_prompts_df = pl.read_parquet(data_config.train_prompts_file_path)
        
    #filter that down to the chosen path, create the dataset and dataloader
    prompt_dataset = PromptDataset(
        all_prompts_df = all_prompts_df,
        chosen_path =  data_config.chosen_subset_path,
        n_samples = data_config.n_samples_train
    )
    prompt_loader = DataLoader(
        dataset = prompt_dataset,
        batch_size=data_config.batch_size,
        shuffle=True
    )
    return prompt_loader