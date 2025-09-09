'''
For each dataset, vary hyperparams, store the linear probe (using a generated key name that we store alongside the hyperparams performance), and test it out on the validation set.
'''

# %%
import torch

from prompts_dataset import DataConfig, get_prompt_loader
from activation_extraction import ActivationExtractor, FinalTokenActivationsDataset
from training_primitives import DataSplit
from basic_probe import ProbeConfig, ProbeEvaluator
import uuid
import polars as pl
import random
# %%
print('finished loading imports')
TRAIN_PROMPTS_FILE_PATH = '/workspace/linear-probes-improve/processed_data/train_prompts.parquet'
VALIDATE_PROMPTS_FILE_PATH = '/workspace/linear-probes-improve/processed_data/validate_prompts.parquet'
N_SAMPLES_TRAIN = 1000
LLM_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 8


N_HYPERPARAM_VARIATIONS_PER_SUBSET_PATH = 10

N_EPOCHS_LIST = [1,3,10,30,100,300]
LR_LIST = [1e-6,3e-6,1e-5,3e-5,1e-4]

# Generate 10 randomly chosen pairs
random.seed(42)  # For reproducibility

# %% Load the prompts
class HyperparamVaryer:
    def __init__(self):
        self.hyperparam_pairs = []
        for _ in range(N_HYPERPARAM_VARIATIONS_PER_SUBSET_PATH):
            n_epochs = random.choice(N_EPOCHS_LIST)
            lr = random.choice(LR_LIST)
            self.hyperparam_pairs.append((n_epochs, lr))
        self.subset_paths_list = pl.scan_parquet('/workspace/linear-probes-improve/processed_data/train_prompts.parquet').select(pl.col('path').unique()).collect()['path'].to_list()
        self.experiment_rows = []
    
    def evaluate_subset_path(self, chosen_subset_path: str):
        n_classes = len(
            pl.scan_parquet(TRAIN_PROMPTS_FILE_PATH)
            .filter(pl.col('path').eq(chosen_subset_path))
            .head(1)
            .collect()
            ['shuffled_choices']
            .item()
        )
        
        data_config = DataConfig(
                    train_prompts_file_path = TRAIN_PROMPTS_FILE_PATH,
                    chosen_subset_path = chosen_subset_path,
                    n_samples_train = N_SAMPLES_TRAIN,
                    batch_size = BATCH_SIZE,
                    llm_name = LLM_NAME
        )
        print(f'fetching prompts for {chosen_subset_path}')
        prompt_loader = get_prompt_loader(data_config)
        train_activations_extractor = ActivationExtractor(data_config, prompt_loader)
        validate_activation_extractor = ActivationExtractor(data_config)
        print('extracting activations')
        train_final_tokens_activations_dataset = train_activations_extractor.get_final_token_activations_dataset()
        validate_final_tokens_activations_dataset = validate_activation_extractor.get_final_token_activations_dataset()
        n_layers = train_final_tokens_activations_dataset[0][0].shape[0]
        print(n_layers)

        for n_epochs, lr in self.hyperparam_pairs:
            print(f'running {chosen_subset_path} for {n_epochs} epochs with lr={lr}')
            for layer in range(n_layers):
                print(f'getting probe for layer {layer}')
                train_probe_config = ProbeConfig(
                    n_epochs = n_epochs,
                    lr = lr,
                    layer = layer,
                    n_classes = n_classes,
                    batch_size = BATCH_SIZE,
                )
                logistic_classifier = self.get_logistic_classifier(train_probe_config, train_final_tokens_activations_dataset)
                validate_probe_config = train_probe_config
                
                validate_probe_evaluator = ProbeEvaluator(
                    probe_config = validate_probe_config,
                    final_tokens_activations_dataset = validate_final_tokens_activations_dataset,
                    data_split = DataSplit.VALIDATE,
                    logistic_classifier = logistic_classifier
                )

                validate_accuracy = validate_probe_evaluator.get_accuracy()

                experiment_info = {
                    'data_config': data_config.__dict__,
                    'probe_config': train_probe_config.__dict__,
                    'validate_accuracy': validate_accuracy
                }
                
                probe_name = f"probe_{uuid.uuid4().hex}"

                experiment_info['probe_name'] = probe_name
                # Save the trained probe
                print('saving probe')
                probe_save_path = f"/workspace/linear-probes-improve/experiments/experiment_1_basic_vs_attn_probe/basic_hyperparam_vary/probes/{probe_name}.pt"
                torch.save(logistic_classifier.state_dict(), probe_save_path)
                experiment_info['probe_save_path'] = probe_save_path
                print('saved probe')

                self.experiment_rows.append(experiment_info)

                experiment_df = pl.DataFrame(self.experiment_rows)
                experiment_df.write_parquet('/workspace/linear-probes-improve/experiments/experiment_1_basic_vs_attn_probe/basic_hyperparam_vary/results.parquet')
                print('updated experiment results')

    
    def get_logistic_classifier(self, train_probe_config: ProbeConfig, train_final_tokens_activations_dataset: FinalTokenActivationsDataset):
        train_probe_evaluator = ProbeEvaluator(
                probe_config = train_probe_config,
                data_split= DataSplit.TRAIN,
                final_tokens_activations_dataset=train_final_tokens_activations_dataset
        )
        train_probe_evaluator.train_logistic_classifier()
        return train_probe_evaluator.logistic_classifier
    
    def evaluate_all(self):
        for chosen_subset_path in self.subset_paths_list:
            self.evaluate_subset_path(chosen_subset_path)

# %%
hyperparam_varyer = HyperparamVaryer()
print('initialised')
hyperparam_varyer.evaluate_all()
# %%
