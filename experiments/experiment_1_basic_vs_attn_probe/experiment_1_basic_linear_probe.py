'''
In this experiment we test:
Can we train a linear probe to predict the correct target class over multiple classes?
Then, can we train a simple attention probe to beat it (random initialisation)?

i.e. use logistic regression on the final token, varying layers to see if we can get out the correct choice (expect this to be a subset of the times that the model knows the correct choice)

similar to what was done in 
https://arxiv.org/pdf/2502.16681
'''

f'''
Plan:
Simple probe:
* use boilerplate linear probe code over -1th token, varying layers. Different model for each dataset, different n_classes.
* Vary hyperparams and optimise over validation (lr, n_epochs, layer)
* pick set of hyperparams for test

Attention probe:
* rather than pred = f(resid_-1^l), we have pred = f_QKV(resid_[0:]^l) where Q K and V are all learnt (Q applies to final token, K to all, V projects to dim n_classes, is what was previously the linear probe)
* vary hyperparams and optimise over validation (lr, n_epochs, layer, d_head, n_heads)

Evaluation:
* we are training on the cross entropy loss. We are measuring the accuracy on the test set
'''
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

# %% config classes
@dataclass
class DataConfig:
    train_prompts_file_path: str
    chosen_subset_path: str
    n_samples_train: int
    batch_size: int
    llm_name: str

class DataSplit(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"

@dataclass
class ProbeConfig:
    n_epochs: int
    lr: float
    layer: int # TODO: also allow None to train one for each layer
    n_classes: int
    batch_size: int # this is the batch size for logistic probe training and eval

# %% final token activations classes

class PromptDataset(Dataset):
    def __init__(self, all_prompts_df: pl.DataFrame, chosen_path: str, n_samples: int):
        self.prompts_df = all_prompts_df.filter(pl.col('path').eq(chosen_path)).head(n_samples)
    
    def __len__(self):
        return len(self.prompts_df)

    def __getitem__(self, idx: int):
        row = self.prompts_df[idx]
        X = row['prompt'].item()
        y = row['shuffled_correct_choice_idx'].item()
        return (X,y)

class FinalTokenActivationsDataset(Dataset):
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
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        # load parquet from disk
        all_prompts_df = pl.read_parquet(self.data_config.train_prompts_file_path)
        
        #filter that down to the chosen path, create the dataset and dataloader
        self.prompt_dataset = PromptDataset(
            all_prompts_df = all_prompts_df,
            chosen_path =  self.data_config.chosen_subset_path,
            n_samples = self.data_config.n_samples_train
        )
        self.prompt_loader = DataLoader(
            dataset = self.prompt_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True
        )

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
# %% logistic regression classes
class LogisticClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
    
    def forward(self,x: torch.Tensor):
        return self.lin1(x)

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


# %%

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
