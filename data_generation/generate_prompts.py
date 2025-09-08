# %%
import polars as pl
import random
from typing import List
# %%
all_qns_df = pl.read_parquet('/workspace/linear-probes-improve/processed_data/choices.parquet')

alphabet = [chr(ord("A") + i) for i in range(26)]
# %%
all_qns_df.filter(pl.col('correct_choice_idx').ge(pl.col('choices').list.len())).select(pl.col('path').unique()).to_dicts()
# %%
def choices_to_str(choices: List[str], alphabet: List[str] = alphabet) -> str:

    out = "Choices: \n"
    for i, choice in enumerate(choices):
        out += f'({alphabet[i]}) {choice} \n'
    out += 'Answer:'
    return out
# %%
all_qns_df = all_qns_df.with_columns(
    shuffled_data = pl.struct(['choices', 'correct_choice_idx']).map_elements(
        lambda x: (lambda shuffled_pairs: {
            'shuffled_choices': [pair[1] for pair in shuffled_pairs],
            'new_correct_idx': [pair[0] for pair in shuffled_pairs].index(x['correct_choice_idx'])
        })(sorted(enumerate(x['choices']), key=lambda _: random.random())),
        return_dtype=pl.Struct([
            pl.Field('shuffled_choices', pl.List(pl.String)),
            pl.Field('new_correct_idx', pl.Int64)
        ])
    )
).with_columns(
    shuffled_choices = pl.col('shuffled_data').struct.field('shuffled_choices'),
    shuffled_correct_choice_idx = pl.col('shuffled_data').struct.field('new_correct_idx')
).drop('shuffled_data')
# %%
prompt_df = all_qns_df.with_columns(
    (pl.col('question') + pl.lit('\n') + pl.col('shuffled_choices').map_elements(choices_to_str)).alias('prompt')
)
# %%
prompt_df.head(1)['prompt'].item()
# %%
prompt_df.write_parquet('/workspace/linear-probes-improve/processed_data/prompts.parquet')
# %%
