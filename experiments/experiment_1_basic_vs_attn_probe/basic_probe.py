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
from activation_extraction import FinalTokenActivationsDataset, ActivationExtractor
from training_primitives import DataSplit, LogisticClassifier
# %%
@dataclass
class ProbeConfig:
    n_epochs: int
    lr: float
    layer: int # TODO: also allow None to train one for each layer
    n_classes: int
    batch_size: int # this is the batch size for logistic probe training and eval




class ProbeEvaluator:
    def __init__(self, probe_config: ProbeConfig, final_tokens_activations_dataset: FinalTokenActivationsDataset, data_split: DataSplit, logistic_classifier: Optional[LogisticClassifier] = None):
        self.probe_config = probe_config
        
        self.final_tokens_activations_loader = DataLoader(dataset=final_tokens_activations_dataset, batch_size = probe_config.batch_size)
        self.data_split = data_split
        h = self.final_tokens_activations_loader.dataset.activations.shape[-1]
        if logistic_classifier is None:
            self.logistic_classifier = LogisticClassifier(h, probe_config.n_classes).to('cuda')
        else:
            self.logistic_classifier = logistic_classifier
        self.optim = Adam(self.logistic_classifier.parameters(),lr=self.probe_config.lr)
        self.loss_fn = CrossEntropyLoss()

    def get_accuracy(self):
        self.logistic_classifier.eval()
        n_correct = 0
        n_total = 0
        for X, actual in self.final_tokens_activations_loader:
            preds = self.logistic_classifier(X[:,self.probe_config.layer,:].to('cuda')).argmax(-1)
            n_correct += ((preds == actual.to('cuda')).sum()).cpu().item()
            n_total += len(preds)
        accuracy = n_correct/n_total
        return accuracy

    def train_logistic_classifier(self):
        assert self.data_split == DataSplit.TRAIN

        layer = self.probe_config.layer
        for i in range(self.probe_config.n_epochs):
            self.logistic_classifier.train()
            self.logistic_classifier.zero_grad()
            for X, actual in self.final_tokens_activations_loader:
                preds = self.logistic_classifier(X[:,layer,:].to('cuda'))
                loss = self.loss_fn(preds, actual.to('cuda')).type(torch.float32).to('cuda')
                loss.backward()
                self.optim.step()
            accuracy = self.get_accuracy()
            print(f'accuracy: {accuracy}')



data_config = DataConfig(
    train_prompts_file_path = '/workspace/linear-probes-improve/processed_data/train_prompts.parquet',
    chosen_subset_path = '/workspace/linear-probes-improve/raw_data/105_click_bait',
    n_samples_train = 1000,
    batch_size = 8,
    llm_name = "Qwen/Qwen2.5-1.5B"
)


# %%
train_activations_extractor = ActivationExtractor(data_config)
train_final_tokens_activations_dataset = train_activations_extractor.get_final_token_activations_dataset()
# %%
train_probe_config = ProbeConfig(
    n_epochs = 2,
    lr = 1e-5,
    layer = -1,
    n_classes = 2,
    batch_size = 8,
)
train_probe_evaluator = ProbeEvaluator(
    probe_config = train_probe_config,
    data_split= DataSplit.TRAIN,
    final_tokens_activations_dataset=train_final_tokens_activations_dataset
)
# %%
train_probe_evaluator.train_logistic_classifier()
# %%
validate_data_config = DataConfig(
    train_prompts_file_path = '/workspace/linear-probes-improve/processed_data/validate_prompts.parquet',
    chosen_subset_path = '/workspace/linear-probes-improve/raw_data/105_click_bait',
    n_samples_train = 100,
    batch_size = 8,
    llm_name = "Qwen/Qwen2.5-1.5B"
)


# %%
validate_probe_config = train_probe_config
validate_activation_extractor = ActivationExtractor(data_config)
validate_final_tokens_activations_dataset = validate_activation_extractor.get_final_token_activations_dataset()
validate_probe_evaluator = ProbeEvaluator(
    probe_config = validate_probe_config,
    final_tokens_activations_dataset = validate_final_tokens_activations_dataset,
    data_split = DataSplit.VALIDATE,
    logistic_classifier = train_probe_evaluator.logistic_classifier
)

validate_accuracy = validate_probe_evaluator.get_accuracy()

print(validate_accuracy)
# %%
