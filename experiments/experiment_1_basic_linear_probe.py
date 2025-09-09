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

from typing import List, Tuple

# %% config classes
@dataclass
class DataConfig:
    train_prompts_file_path: str
    chosen_subset_path: str
    n_samples_train: int
    batch_size: int
    llm_name: str

@dataclass
class ProbeConfig:
    n_epochs: int
    lr: float
    layer: int #also allow None to train one for each layer
    n_classes: int

# %% final token activations classes

class PromptDataset(Dataset):
    def __init__(self, all_train_prompts_df: pl.DataFrame, chosen_path: str, n_samples: int):
        self.train_prompts_df = all_train_prompts_df.filter(pl.col('path').eq(chosen_path)).head(n_samples)
    
    def __len__(self):
        return len(self.train_prompts_df)

    def __getitem__(self, idx: int):
        row = self.train_prompts_df[idx]
        X = row['prompt'].item()
        y = row['shuffled_correct_choice_idx'].item()
        return (X,y)

class FinalTokenActivations(Dataset):
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
        all_train_prompts_df = pl.read_parquet(self.data_config.train_prompts_file_path)
        
        #filter that down to the chosen path, create the dataset and dataloader
        self.train_prompt_dataset = PromptDataset(
            all_train_prompts_df = all_train_prompts_df,
            chosen_path =  self.data_config.chosen_subset_path,
            n_samples = self.data_config.n_samples_train
        )
        self.train_prompt_loader = DataLoader(
            dataset = self.train_prompt_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True
        )

        #load the llm from huggingface via nnisght wrapper
        self.llm = LanguageModel(self.data_config.llm_name, device_map = "auto", dispatch=True)

    def get_final_token_activations_dataset(self):
        final_token_activations_pairs=[]
        self.llm.eval()
        for X, y in self.train_prompt_loader:

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

        final_token_activations_dataset = FinalTokenActivations(final_token_activations_pairs)
        return final_token_activations_dataset
# %% logistic regression classes
class LogisticClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
    
    def forward(self,x: torch.Tensor):
        return self.lin1(x)

class ProbeTrainer:
    def __init__(self, probe_config: ProbeConfig, train_final_tokens_activations_loader: DataLoader):
        self.probe_config = probe_config
        self.train_final_tokens_activations_loader = train_final_tokens_activations_loader
        h = self.train_final_tokens_activations_loader.dataset.activations.shape[-1]
        self.logistic_classifier = LogisticClassifier(h, probe_config.n_classes).to('cuda')
        self.optim = Adam(self.logistic_classifier.parameters(),lr=self.probe_config.lr)
        self.loss_fn = CrossEntropyLoss()

    def get_train_accuracy(self):
        self.logistic_classifier.eval()
        n_correct = 0
        n_total = 0
        for X, actual in self.train_final_tokens_activations_loader:
            preds = self.logistic_classifier(X[:,self.probe_config.layer,:].to('cuda')).argmax(-1)
            n_correct += ((preds == actual.to('cuda')).sum()).cpu().item()
            n_total += len(preds)
        accuracy = n_correct/n_total
        return accuracy

    def train_logistic_classifier(self):
        layer = self.probe_config.layer
        for i in range(self.probe_config.n_epochs):
            self.logistic_classifier.train()
            self.logistic_classifier.zero_grad()
            for X, actual in self.train_final_tokens_activations_loader:
                preds = self.logistic_classifier(X[:,layer,:].to('cuda'))
                loss = self.loss_fn(preds, actual.to('cuda')).type(torch.float32).to('cuda')
                loss.backward()
                self.optim.step()
            accuracy = self.get_train_accuracy()
            print(f'accuracy: {accuracy}')


# %%

data_config = DataConfig(
    train_prompts_file_path = '/workspace/linear-probes-improve/processed_data/train_prompts.parquet',
    chosen_subset_path = '/workspace/linear-probes-improve/raw_data/105_click_bait',
    n_samples_train = 100,
    batch_size = 8,
    llm_name = "Qwen/Qwen2.5-1.5B"
)

probe_config = ProbeConfig(
    n_epochs = 50,
    lr = 1e-4,
    layer = -1,
    n_classes = 2,
)
# %%
activations_extractor = ActivationExtractor(data_config)
train_final_tokens_activations = activations_extractor.get_final_token_activations_dataset()
train_final_tokens_activations_loader = DataLoader(train_final_tokens_activations, batch_size = 8)
# %%
probe_trainer = ProbeTrainer(
    probe_config = probe_config,
    train_final_tokens_activations_loader=train_final_tokens_activations_loader
)

probe_trainer.train_logistic_classifier()
# %%