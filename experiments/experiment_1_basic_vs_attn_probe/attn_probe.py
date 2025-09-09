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
class AttnProbeConfig:
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
        assert self.n_heads * self.d_head == self.d_model, 'need n_heads * d_heads = d_model'
        
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

    
class AttentionProbeEvaluator:
    def __init__(self, probe_config: AttnProbeConfig, all_tokens_activations_dataset: AllTokenActivationsDataset, data_split: DataSplit, attention_probe_classifier: Optional[AttentionProbeClassifier] = None):
        self.probe_config = probe_config
        
        self.all_tokens_activations_loader = DataLoader(dataset=all_tokens_activations_dataset, batch_size=probe_config.batch_size)
        self.data_split = data_split
        
        # Get dimensions from the dataset by directly accessing the stored data
        # The AllTokenActivationsDataset stores pairs with shape (1, n_layers, seq_len, d_model)
        sample_activations_pair = all_tokens_activations_dataset.all_token_activation_pairs[0]
        sample_activations = sample_activations_pair[0]  # (1, n_layers, seq_len, d_model)
        d_model = sample_activations.shape[-1]
        
        if attention_probe_classifier is None:
            self.attention_probe_classifier = AttentionProbeClassifier(d_model, probe_config.d_head, probe_config.n_heads, probe_config.n_classes).to('cuda')
        else:
            self.attention_probe_classifier = attention_probe_classifier
        
        self.optim = Adam(self.attention_probe_classifier.parameters(), lr=self.probe_config.lr)
        self.loss_fn = CrossEntropyLoss()

    def get_accuracy(self):
        self.attention_probe_classifier.eval()
        n_correct = 0
        n_total = 0
        for activations, actual, attention_mask in self.all_tokens_activations_loader:
            # activations shape: (batch_size, n_layers, seq_len, d_model)
            # Select the specified layer
            layer_activations = activations[:, self.probe_config.layer, :, :].to('cuda')
            preds = self.attention_probe_classifier(layer_activations, attention_mask.to('cuda')).argmax(-1)
            n_correct += ((preds == actual.to('cuda')).sum()).cpu().item()
            n_total += len(preds)
        accuracy = n_correct / n_total
        return accuracy

    def train_attention_probe_classifier(self):
        assert self.data_split == DataSplit.TRAIN

        layer = self.probe_config.layer
        for i in range(self.probe_config.n_epochs):
            self.attention_probe_classifier.train()
            self.attention_probe_classifier.zero_grad()
            for activations, actual, attention_mask in self.all_tokens_activations_loader:
                # activations shape: (batch_size, n_layers, seq_len, d_model)
                # Select the specified layer
                layer_activations = activations[:, layer, :, :].to('cuda')
                print(activations.shape)
                preds = self.attention_probe_classifier(layer_activations, attention_mask.to('cuda'))
                loss = self.loss_fn(preds, actual.to('cuda')).type(torch.float32).to('cuda')
                loss.backward()
                self.optim.step()
            accuracy = self.get_accuracy()
            print(f'accuracy: {accuracy}')


# %%
if __name__ == "__main__":
    '''
    Component test for training and evaluating attention probe on validation set
    '''
    data_config = DataConfig(
        train_prompts_file_path = '/workspace/linear-probes-improve/processed_data/train_prompts.parquet',
        chosen_subset_path = '/workspace/linear-probes-improve/raw_data/105_click_bait',
        n_samples_train = 10,
        batch_size = 1,
        llm_name = "Qwen/Qwen2.5-1.5B"
    )
    
    prompt_loader = get_prompt_loader(data_config)
    train_activations_extractor = ActivationExtractor(data_config, prompt_loader)
    train_all_tokens_activations_dataset = train_activations_extractor.get_all_token_activations_dataset() 
    train_probe_config = AttnProbeConfig(
        n_epochs = 2,
        lr = 1e-5,
        layer = -1,
        n_classes = 2,
        batch_size = 2,
        n_heads = 24,  # Number of attention heads for Qwen2.5-1.5B
        d_head = 64,   # Dimension per head for Qwen2.5-1.5B
    )
    train_probe_evaluator = AttentionProbeEvaluator(
        probe_config = train_probe_config,
        data_split= DataSplit.TRAIN,
        all_tokens_activations_dataset=train_all_tokens_activations_dataset
    )
    train_probe_evaluator.train_attention_probe_classifier()
    validate_data_config = DataConfig(
        train_prompts_file_path = '/workspace/linear-probes-improve/processed_data/validate_prompts.parquet',
        chosen_subset_path = '/workspace/linear-probes-improve/raw_data/105_click_bait',
        n_samples_train = 100,
        batch_size = 1,
        llm_name = "Qwen/Qwen2.5-1.5B"
    )

    validate_probe_config = train_probe_config
    validate_activation_extractor = ActivationExtractor(validate_data_config)
    validate_all_tokens_activations_dataset = validate_activation_extractor.get_all_token_activations_dataset()
    validate_probe_evaluator = AttentionProbeEvaluator(
        probe_config = validate_probe_config,
        all_tokens_activations_dataset = validate_all_tokens_activations_dataset,
        data_split = DataSplit.VALIDATE,
        attention_probe_classifier = train_probe_evaluator.attention_probe_classifier
    )

    validate_accuracy = validate_probe_evaluator.get_accuracy()

    print(f'validation accuracy {validate_accuracy}')

# %%
