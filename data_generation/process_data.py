'''
This script will process the datasets (union train validate test) and put these into 1 dataframe for each dataset with Schema [("question", String), ("choices", List[String]), ("correct_choice_idx", Int32)], where correct_choice is the idx in the list

From these, we will construct prompts (varying formats). 

If a model produces a generative response, then we use a model grader at a later stage to parse is_correct True/False.
'''
# %% imports
import polars as pl
import datasets
import json

from dataclasses import dataclass
from typing import Callable
# %% explore a hf dataset
# Load and explore the sciq dataset
dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/47_reasoning_tf')

df = pl.from_pandas(dataset['train'].to_pandas())
df
# %%

def load_hf_dataset(path: str, split: str = 'train') -> pl.DataFrame:
    """Generic function to load any HuggingFace dataset from disk"""
    dataset = datasets.load_from_disk(path)
    df = pl.from_pandas(dataset[split].to_pandas())
    return df

sciq_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/36_sciq_tf')

def process_sciq_tf(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = row['question']
        out_row['choices'] = [row['distractor1'], row['distractor2'], row['distractor3'], row['correct_answer']]
        out_row['correct_choice_idx'] = len(out_row['choices']) -1
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_sciq_tf(sciq_df)
    
# %%

reasoning_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/47_reasoning_tf')

def process_reasoning_tf(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = row['question']
        out_row['choices'] = row['choices']['text']
        out_row['correct_choice_idx'] = row['choices']['label'].index(row['answerKey'])
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_reasoning_tf(reasoning_df)

# %%
cs_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/54_cs_tf')
# %%
def process_cs(df: pl.DataFrame) -> pl.DataFrame:
    df1 = df.filter(pl.col('confidence').ge(0.9))
    out_data = []
    for row in df1.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = row['question']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = out_row['choices'].index(row['answer'])
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_cs(cs_df)

# %%
cola_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/87_glue_cola')
# %%
def process_cola(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = row['sentence'] + '\n Is this grammatically correct?'
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['label']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_cola(cola_df)
# %%
mrpc_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/89_glue_mrpc')
# %%
def process_mrpc(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = row['sentence1']+ '\n \n' + row['sentence1'] + '\n Are the above 2 sentences equivalent?'
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['label']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_mrpc(mrpc_df)
# %%
qnli_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/90_glue_qnli')
# %%
def process_qnli(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Does the answer to the following question lie in the text below it \n \n' +row['question']+ '\n' + row['sentence']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['label']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_qnli(qnli_df)
# %%
qqp_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/91_glue_qqp')
# %%
def process_qqp(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = row['question1']+ '\n \n' + row['question2'] + '\n Are the above 2 questions equivalent?'
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['label']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_qqp(qqp_df)
# %%
sst2_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/92_glue_sst2')
# %%
def process_sst2(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this positive sentiment? \n ' + row['sentence']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['label']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_sst2(sst2_df)
# %%
def load_aigen(path: str, split: str = 'train') -> pl.DataFrame:
    aigen_dataset = datasets.load_from_disk(path)

    aigen_df = pl.from_pandas(aigen_dataset[split].to_pandas()).head(10000)
    return aigen_df

aigen_df = load_aigen('/workspace/linear-probes-improve/raw_data/94_ai_gen')
# %%
def process_aigen(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Was this generated by AI? \n ' + row['text']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['generated']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_aigen(aigen_df)
# %%
toxic_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/95_toxic_is')
# %%
def process_toxic(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this prompt toxic? \n ' + row['user_input']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['human_annotation']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_toxic(toxic_df)
# %%
clickbait_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/105_click_bait')

# %%
def process_clickbait(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this clickbait? \n ' + row['text']
        out_row['choices'] = ['clickbait','factual']
        out_row['correct_choice_idx'] = out_row['choices'].index(row['label'])
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_clickbait(clickbait_df)
# %%

def load_aimade(path: str) -> pl.DataFrame:
    aimade_df = pl.read_csv(path).head(10000)
    return aimade_df

aimade_df = load_aimade('/workspace/linear-probes-improve/raw_data/110_aimade_humangpt3/FinalDataset.csv')
aimade_df
# %%
def process_aimade(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this human or gpt3? \n ' + row['sentence']
        out_row['choices'] = ['human','gpt3']
        out_row['correct_choice_idx'] = row['class']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_aimade(aimade_df)
# %%
movie_sent_df = load_hf_dataset('/workspace/linear-probes-improve/raw_data/113_movie_sent')

# %%
def process_movie(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this review positive? \n ' + row['text']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['label']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_movie(movie_sent_df)
# %%
def load_headline_trump(path: str) -> pl.DataFrame:
    headline_trump = pl.read_csv(path)[['headline','is_trump']]
    return headline_trump

headline_trump = load_headline_trump('/workspace/linear-probes-improve/raw_data/21_headline_istrump.csv')
headline_trump
# %%
def process_headline(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this headline trump? \n ' + row['headline']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = 1-row['is_trump']
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_headline(headline_trump)
# %%

def load_phys(path: str) -> pl.DataFrame:
    with open(path, 'r') as f:
        phys_data = [json.loads(line) for line in f]
    phys_df = pl.DataFrame(phys_data)
    return phys_df

phys_df = load_phys('/workspace/linear-probes-improve/raw_data/44_phys_tf.jsonl')
# %%
def process_phys(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Which option is more reasonable for the following goal? \n ' + row['goal']
        out_row['choices'] = [row['sol1'],row['sol2']]
        out_row['correct_choice_idx'] = 0
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_phys(phys_df)
# %%
def load_spam(path: str) -> pl.DataFrame:
    spam_df = pl.read_csv(path)
    return spam_df

spam_df = load_spam('/workspace/linear-probes-improve/raw_data/96_spam_is.csv')
spam_df
# %%
def process_spam(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this ham or spam? \n ' + row['Message']
        out_row['choices'] = ['ham', 'spam']
        out_row['correct_choice_idx'] = out_row['choices'].index(row['Category'])
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_spam(spam_df)
# %%
def load_hate(path: str) -> pl.DataFrame:
    hate_df = pl.read_csv(path)
    return hate_df

hate_df = load_hate('/workspace/linear-probes-improve/raw_data/106_hate_hate.csv')
# %%
def process_hate(df: pl.DataFrame) -> pl.DataFrame:
    out_data = []
    for row in df.iter_rows(named=True):
        out_row = dict()
        out_row['question'] = 'Is this hate speech? \n ' + row['tweet']
        out_row['choices'] = ['yes','no']
        out_row['correct_choice_idx'] = int(row['hate_speech'] > 3)
        out_data.append(out_row)
    return pl.DataFrame(out_data)

process_hate(hate_df)['correct_choice_idx']
# %%
@dataclass
class DatasetConfig:
    path: str
    load_fn: Callable
    process_fn: Callable


# %%
dataset_configs = [
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/36_sciq_tf',
        load_fn = load_hf_dataset,
        process_fn = process_sciq_tf
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/47_reasoning_tf',
        load_fn = load_hf_dataset,
        process_fn = process_reasoning_tf
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/54_cs_tf',
        load_fn = load_hf_dataset,
        process_fn = process_cs
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/87_glue_cola',
        load_fn = load_hf_dataset,
        process_fn = process_cola
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/89_glue_mrpc',
        load_fn = load_hf_dataset,
        process_fn = process_mrpc
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/90_glue_qnli',
        load_fn = load_hf_dataset,
        process_fn = process_qnli
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/91_glue_qqp',
        load_fn = load_hf_dataset,
        process_fn = process_qqp
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/92_glue_sst2',
        load_fn = load_hf_dataset,
        process_fn = process_sst2
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/94_ai_gen',
        load_fn = load_aigen,
        process_fn = process_aigen
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/95_toxic_is',
        load_fn = load_hf_dataset,
        process_fn = process_toxic
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/105_click_bait',
        load_fn = load_hf_dataset,
        process_fn = process_clickbait
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/110_aimade_humangpt3/FinalDataset.csv',
        load_fn = load_aimade,
        process_fn = process_aimade
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/113_movie_sent',
        load_fn = load_hf_dataset,
        process_fn = process_movie
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/21_headline_istrump.csv',
        load_fn = load_headline_trump,
        process_fn = process_headline
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/44_phys_tf.jsonl',
        load_fn = load_phys,
        process_fn = process_phys
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/96_spam_is.csv',
        load_fn = load_spam,
        process_fn = process_spam
    ),
    DatasetConfig(
        path='/workspace/linear-probes-improve/raw_data/106_hate_hate.csv',
        load_fn = load_hate,
        process_fn = process_hate
    ),
]
# %%
def load_and_process(config: DatasetConfig, split = None) -> pl.DataFrame:
    path = config.path
    load_fn = config.load_fn
    process_fn = config.process_fn

    if split is None:
        loaded_df = load_fn(path)
    else:
        loaded_df = load_fn(path,split)
    processed_df = process_fn(loaded_df)
    return processed_df

processed_dfs = []

for dataset_config in dataset_configs:
    if dataset_config.load_fn == load_hf_dataset:
        for split in ['train','validate','test']:
            try:
                df = load_and_process(dataset_config,split).with_columns(path = pl.lit(dataset_config.path))
            except:
                print(f'failed for {dataset_config.path}, {split}')
            processed_dfs.append(df)
    else:
        df = load_and_process(dataset_config).with_columns(path = pl.lit(dataset_config.path))
        processed_dfs.append(df)
# %%
processed_data = pl.concat(processed_dfs)
processed_data.write_parquet('/workspace/linear-probes-improve/processed_data/choices.parquet')
# %%
