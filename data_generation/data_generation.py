# %% imports
import pandas as pd
import polars as pl

import requests
import os

from datasets import load_dataset
import kagglehub
# %% load master csv
df = pl.read_csv('/workspace/linear-probes-improve/probing_datasets_MASTER.csv')
df
# %%
binary_datasets_df = df.filter(pl.col('Data type').eq('Binary Classification') & pl.col('Probe to').is_not_null() & pl.col('Link').is_not_null() & pl.col('Link').ne('^'))
binary_datasets_df
# %%
for row in binary_datasets_df.sort('Link').iter_rows(named=True):
    print(row['Link'], row['Dataset save name'])
# %% Download and save datasets to raw_data
datasets_to_download = [
    ("allenai/quartz", "../raw_data/47_reasoning_tf"),
    ("allenai/sciq", "../raw_data/36_sciq_tf"),
    ("andythetechnerd03/AI-human-text", "../raw_data/94_ai_gen"),
    ("cornell-movie-review-data/rotten_tomatoes", "../raw_data/113_movie_sent"),
    ("lmsys/toxic-chat", "../raw_data/95_toxic_is", 'toxicchat0124'),
    ("nyu-mll/glue", "../raw_data/87_glue_cola", "cola"),
    ("nyu-mll/glue", "../raw_data/89_glue_mrpc", "mrpc"),
    ("nyu-mll/glue", "../raw_data/90_glue_qnli", "qnli"),
    ("nyu-mll/glue", "../raw_data/91_glue_qqp", "qqp"),
    ("nyu-mll/glue", "../raw_data/92_glue_sst2", "sst2"),
    ("rkotari/clickbait", "../raw_data/105_click_bait"),
    ("tasksource/commonsense_qa_2.0", "../raw_data/54_cs_tf"),
    ("truthfulqa/truthful_qa", "../raw_data/41_truthqa_tf", 'multiple_choice')
]

for dataset_info in datasets_to_download:
    dataset_name = dataset_info[0]
    save_path = dataset_info[1]
    
    try:
        if len(dataset_info) == 3:  # GLUE datasets with config
            config = dataset_info[2]
            ds = load_dataset(dataset_name, config)
        else:
            ds = load_dataset(dataset_name)
        
        # Save the dataset
        ds.save_to_disk(save_path)
        print(f"Saved {dataset_name} to {save_path}")
        
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
# %% Download additional datasets from URLs
url_datasets_to_download = [
    ("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv", "../raw_data/106_hate_hate.csv"),
    ("https://raw.githubusercontent.com/wesg52/world-models/refs/heads/main/data/entity_datasets/headline.csv", "../raw_data/21_headline_istrump.csv"),
    #("https://www.dropbox.com/scl/fo/14oxabm2eq47bkw2u0oxo/AKFTcikvAB8-GVdoBztQHxE?rlkey=u9qny1tsza6lqetzzua3jr8xn&e=1&dl=1", "../raw_data/56_wikidatasex_or_gender.csv"), #note this one is v big 2GB
    ("https://github.com/ybisk/ybisk.github.io/raw/master/piqa/data/train.jsonl", "../raw_data/44_phys_tf.jsonl"),
    ("https://github.com/CogComp/MCTACO/raw/master/dataset/dev_3783.tsv", "../raw_data/42_temp_sense.tsv"),
    ("https://github.com/HarshOza36/Detection-Of-Machine-Generated-Text/raw/master/data/Final%20Dataset.zip", "../raw_data/110_aimade_humangpt3.zip")
]

for url_info in url_datasets_to_download:
    url = url_info[0]
    save_path = url_info[1]
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {url} to {save_path}")
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# %% Download Kaggle datasets

kagglehub.login()
kaggle_datasets = [
    ("clmentbisaillon/fake-and-real-news-dataset", "/workspace/linear-probes-improve/raw_data/100_news_fake.csv"),
    ("team-ai/spam-text-message-classification", "/workspace/linear-probes-improve/raw_data/96_spam_is.csv")
]

for dataset_info in kaggle_datasets:
    dataset_name = dataset_info[0]
    save_path = dataset_info[1]
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download latest version
        path = kagglehub.dataset_download(dataset_name)
        print(f"Downloaded {dataset_name} to temporary path: {path}")
        
        # Copy files from temporary kagglehub cache to desired location
        import shutil
        if os.path.isdir(path):
            # If the downloaded path is a directory, copy all files
            for file in os.listdir(path):
                src_file = os.path.join(path, file)
                if os.path.isfile(src_file):
                    # For specific dataset expectations, adjust filename as needed
                    if dataset_name == "clmentbisaillon/fake-and-real-news-dataset":
                        if file.endswith('.csv'):
                            dst_file = save_path
                            shutil.copy2(src_file, dst_file)
                            print(f"Copied {src_file} to {dst_file}")
                            break
                    elif dataset_name == "team-ai/spam-text-message-classification":
                        if file.endswith('.csv'):
                            dst_file = save_path
                            shutil.copy2(src_file, dst_file)
                            print(f"Copied {src_file} to {dst_file}")
                            break
        else:
            # If the downloaded path is a file, copy it directly
            shutil.copy2(path, save_path)
            print(f"Copied {path} to {save_path}")
        
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")

# %% b21212a54385872a8164e946ab3b8d9f

# Copy files from temporary kagglehub cache to desired locations
 '''
Downloaded clmentbisaillon/fake-and-real-news-dataset to temporary path: /root/.cache/kagglehub/datasets/clmentbisaillon/fake-and-real-news-dataset/versions/1
Downloaded team-ai/spam-text-message-classification to temporary path: /root/.cache/kagglehub/datasets/team-ai/spam-text-message-classification/versions/1
'''
 