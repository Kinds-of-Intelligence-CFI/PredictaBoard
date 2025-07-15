import glob
import os

import pandas as pd
from dotenv import load_dotenv
from utils.embedding import extract_openai_embeddings


def save_processed(filename, results_df, compression=False, compression_kwargs=None):
    """
    Save the processed DataFrame to a json file.
    """
    if not compression:
        compression_kwargs = None
    elif compression and compression_kwargs is None:
        compression_kwargs = {"method": "gzip", "compresslevel": 1, "mtime": 1}

    results_df.to_json(
        filename, orient="columns", indent=1, compression=compression_kwargs
    )

embedding_model = "text-embedding-3-large"
batch_size = 2048
sleep_time_if_fail = 0.1

# load the OpenAI API key
load_dotenv("../../.env")

data_folder_location = os.path.join("../../data", "open-llm-leaderboard-v2")
results_folder_location = os.path.join(data_folder_location, "computed_embeddings")

# the first line if you want to only run the MMLU Pro prompts, otherwise it runs all of them
# prompt_files = sorted(glob.glob(os.path.join(data_folder_location, "mmlu_pro_prompts.csv")))
prompt_files = sorted(glob.glob(os.path.join(data_folder_location, "*_prompts.csv")))

prompts_dfs = [pd.read_csv(prompt) for prompt in prompt_files]

# remove the directory from filenames
prompt_files = [prompt.split("/")[-1] for prompt in prompt_files]

for filename, prompts_df in zip(prompt_files, prompts_dfs):
    print(f"File: {filename}")

    print("Computing embeddings...")
    # extract the embeddings
    embeddings_list = extract_openai_embeddings(prompts_df, embedding_model, batch_size, sleep_time_if_fail,
                                                col_name="prompt")
    print("Done computing embeddings.")
    # now add them as a new column
    prompts_df["openai_embeddings"] = embeddings_list

    # save the new dataframe
    results_filename = filename.replace("prompts", "prompts_with_openai_embeddings").replace("csv", "json")
    print(results_filename)
    save_processed(os.path.join(results_folder_location, results_filename), prompts_df, compression=False)
