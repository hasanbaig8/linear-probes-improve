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
from activation_extraction import AllTokenActivationsDataset, ActivationExtractor
from training_primitives import DataSplit, LogisticClassifier
# %%
@dataclass
class ProbeConfig:
    n_epochs: int
    lr: float
    layer: int # TODO: also allow None to train one for each layer
    n_classes: int
    batch_size: int # this is the batch size for logistic probe training and eval
    d_head: int
    n_heads: int

# %%
class AttentionProbeClassifier(nn.Module):
    def __init__(self, input_dim: int, d_head: int, n_heads: int, output_dim: int):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.d_model = input_dim
        assert self.n_heads * self.d_heads == self.d_model, 'need n_heads * d_heads = d_model'
        
        # Multi-head attention components
        self.W_q = nn.Linear(input_dim, d_head * n_heads, bias=False)
        self.W_k = nn.Linear(input_dim, d_head * n_heads, bias=False)
        self.W_v = nn.Linear(input_dim, d_head * n_heads, bias=False)
        self.W_o = nn.Linear(d_head * n_heads, input_dim, bias=False)
        
        # Final classification layer
        self.classifier = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Only query from the last token
        q = self.W_q(x[:, -1:, :])  # (batch_size, 1, d_head * n_heads)
        # Keys and values from all tokens
        k = self.W_k(x)  # (batch_size, seq_len, d_head * n_heads)
        v = self.W_v(x)  # (batch_size, seq_len, d_head * n_heads)
        
        # Reshape for multi-head attention using einops
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads, d=self.d_head)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads, d=self.d_head)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads, d=self.d_head)
        
        # Compute attention scores using einsum
        scores = einsum(q, k, 'b h s_q d, b h s_k d -> b h s_q s_k') / (self.d_head ** 0.5)
        
        # Apply attention mask if provided (for variable sequence lengths)
        if attention_mask is not None:
            # attention_mask shape: (batch_size, seq_len)
            # Expand to match scores dimensions
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, n_heads, 1, seq_len)
        
        # Apply attention to values using einsum
        attn_output = einsum(attn_weights, v, 'b h s_q s_k, b h s_k d -> b h s_q d')
        
        # Concatenate heads using einops
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        
        # Apply output projection
        output = self.W_o(attn_output)  # (batch_size, 1, d_model)
        
        # Squeeze the sequence dimension since we only have one query token
        final_token_output = output.squeeze(1)  # (batch_size, d_model)
        
        # Apply classification layer
        logits = self.classifier(final_token_output)  # (batch_size, output_dim)
        
        return logits