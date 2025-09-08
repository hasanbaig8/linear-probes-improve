'''
This script will process the datasets (union train validate test) and put these into 1 dataframe for each dataset with Schema [("question", String), ("choices", List[String]), ("correct_choice_idx", Int32)], where correct_choice is the idx in the list

From these, we will construct prompts (varying formats). 

If a model produces a generative response, then we use a model grader at a later stage to parse is_correct True/False.
'''
# %% imports
import polars as pl
import datasets
# %% explore a hf dataset
# Load and explore the sciq dataset
dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/47_reasoning_tf')

df = pl.from_pandas(dataset['train'].to_pandas())
df
# %%
sciq_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/36_sciq_tf')

sciq_df = pl.from_pandas(sciq_dataset['train'].to_pandas())

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
reasoning_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/47_reasoning_tf')

reasoning_df = pl.from_pandas(reasoning_dataset['train'].to_pandas())

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
cs_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/54_cs_tf')

cs_df = pl.from_pandas(cs_dataset['train'].to_pandas())
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
cola_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/87_glue_cola')

cola_df = pl.from_pandas(cola_dataset['train'].to_pandas())
cola_df[0]['sentence'].item()
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
mrpc_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/89_glue_mrpc')

mrpc_df = pl.from_pandas(mrpc_dataset['train'].to_pandas())
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
qnli_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/90_glue_qnli')

qnli_df = pl.from_pandas(qnli_dataset['train'].to_pandas())
qnli_df
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
qqp_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/91_glue_qqp')

qqp_df = pl.from_pandas(qqp_dataset['train'].to_pandas())
qqp_df
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
sst2_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/92_glue_sst2')

sst2_df = pl.from_pandas(sst2_dataset['train'].to_pandas())
sst2_df
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
aigen_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/94_ai_gen')

aigen_df = pl.from_pandas(aigen_dataset['train'].to_pandas()).head(10000)
aigen_df[1]['text'].item()
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
toxic_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/95_toxic_is')

toxic_df = pl.from_pandas(toxic_dataset['train'].to_pandas())
toxic_df
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
clickbait_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/105_click_bait')

clickbait_df = pl.from_pandas(clickbait_dataset['train'].to_pandas())
clickbait_df

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

aimade_df = pl.read_csv('/workspace/linear-probes-improve/raw_data/110_aimade_humangpt3/FinalDataset.csv').head(10000)
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
movie_sent_dataset = datasets.load_from_disk('/workspace/linear-probes-improve/raw_data/113_movie_sent')

movie_sent_df = pl.from_pandas(movie_sent_dataset['train'].to_pandas())
movie_sent_df

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
headline_trump = pl.read_csv('/workspace/linear-probes-improve/raw_data/21_headline_istrump.csv')[['headline','is_trump']]
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
import json

with open('/workspace/linear-probes-improve/raw_data/44_phys_tf.jsonl', 'r') as f:
    phys_data = [json.loads(line) for line in f]
phys_df = pl.DataFrame(phys_data)
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
spam_df = pl.read_csv('/workspace/linear-probes-improve/raw_data/96_spam_is.csv')
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
hate_df = pl.read_csv('/workspace/linear-probes-improve/raw_data/106_hate_hate.csv')
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
