import glob
import os

import pandas as pd
from embeddings_utils.embedding import Embedder
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from warnings import warn

DATA_FOLDER_LOCATION = os.path.join(".", "data")
MMLU_PRO_OPENAI_RESULTS_LOCATION = os.path.join("..", "..", "data", "open-llm-leaderboard-v2", "computed_embeddings", "mmlu_pro_prompts_with_openai_embeddings.json")

def load_csv_pairs(prompt_files: List[str], result_files: List[str]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load pairs of CSV files.
    
    Args:
        prompt_files (List[str]): List of file paths for the prompt CSVs.
        result_files (List[str]): List of file paths for the result CSVs.
    
    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (prompt_df, result_df) pairs.
    """
    assert len(prompt_files) == len(result_files), "Mismatch in number of prompt and result files."
    return [(pd.read_csv(prompt), pd.read_csv(result)) for prompt, result in zip(prompt_files, result_files)]


def merge_and_filter_data(
    prompt_df: pd.DataFrame,
    result_df: pd.DataFrame,
    selected_llm: str
) -> pd.DataFrame:
    """
    Merge prompt and result data, and filter by LLMs if specified.
    
    Args:
        prompt_df (pd.DataFrame): DataFrame with prompts.
        result_df (pd.DataFrame): DataFrame with LLM results.
        selected_llm (str): LLM name to include.
    
    Returns:
        pd.DataFrame: Merged and filtered DataFrame.
    """
    prompt_df["id"] = prompt_df["id"].astype(str)
    result_df.drop_duplicates(subset="model", keep="last", inplace=True) # keep latest time stamp
    merged_df = result_df.melt(id_vars=["model", "timestamp"], var_name="id", value_name="response")
    merged_df = merged_df.merge(prompt_df, left_on="id", right_on="id", how="right")
    
    # Get IDs with NaN responses for specific models - should only apply to MMLU-Pro
    gpt_4o_may_nan = merged_df.loc[
        (merged_df["model"] == "OpenAI__GPT-4o-2024-05-13") & (merged_df["response"].isna()), "id"
    ].tolist()
    gpt_4o_aug_nan = merged_df.loc[
        (merged_df["model"] == "OpenAI__GPT-4o-2024-08-06") & (merged_df["response"].isna()), "id"
    ].tolist()
    gpt_4o_min_nan = merged_df.loc[
        (merged_df["model"] == "OpenAI__GPT-4o-mini") & (merged_df["response"].isna()), "id"
    ].tolist()

    # Combine all NaN-related IDs
    nan_ids = list(set(gpt_4o_may_nan + gpt_4o_aug_nan + gpt_4o_min_nan))

    # Filter for the selected LLM and exclude NaN-related IDs
    merged_df = merged_df[
        (merged_df["model"] == selected_llm) &
        (~merged_df["id"].isin(nan_ids))
        ]
    
    return merged_df

def process(
    prompt_files: List[str],
    result_files: List[str],
    selected_llm: str,
    test_size: float = 0.2,
    seed: int = 1997,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Process multiple benchmarks, merging, filtering, and partitioning them, and store results in a dictionary.
    
    Args:
        prompt_files (List[str]): List of prompt CSV file paths.
        result_files (List[str]): List of result CSV file paths.
        selected_llm (str): LLM name to include.
        test_size (float, optional): Proportion of test data. Defaults to 0.2.
    
    Returns:
        dict: Dictionary where keys are benchmark names and values are a dictionary of train, validation, and test sets.
    """
    csv_pairs = load_csv_pairs(prompt_files, result_files)
    if len(csv_pairs) == 0:
        warn("No result CSVs found for provided LLM")
    results = {}
    for prompt_file, (prompt_df, result_df) in zip(prompt_files, csv_pairs):
        # Extract benchmark name from file path
        benchmark_name = os.path.splitext(os.path.basename(prompt_file))[0].replace("_prompts", "")
        
        # Process data
        merged_data = merge_and_filter_data(prompt_df, result_df, selected_llm)
        try:
            train_df, test_df = train_test_split(merged_data, test_size=test_size, random_state=seed, shuffle = True)
            train_df, val_df = train_test_split(train_df, test_size=test_size/(1-test_size), random_state=seed, shuffle = False)
            # Store in dictionary
            results[benchmark_name] = {
                "train" : train_df,
                "val" : val_df,
                "test" : test_df,
            }
        except:  # noqa: E722
            continue
    if len(results) > 0:
        return results
    else:
        raise RuntimeError("No results found")


def process_benchmarks(selected_llm: str = "migtissera__Tess-v2.5.2-Qwen2-72B", data_folder_location = DATA_FOLDER_LOCATION) -> Dict[str, Dict[str, pd.DataFrame]]:
    prompt_files = sorted(glob.glob(os.path.join(data_folder_location, "open-llm-leaderboard-v2", "*_prompts.csv")))
    results_files = sorted(glob.glob(os.path.join(data_folder_location, "open-llm-leaderboard-v2", "*_results.csv")))
    results_files = [r for r in results_files if os.path.basename(r) != "leaderboard_aggregate_results.csv"]
    results = process(prompt_files, results_files, selected_llm, test_size=0.2, seed=1997)
    return results

# Code below is for reformatting the benchmark result for use in the embeddings assessors notebook # TODO: Unify the schemas we use so we have to do less transformation

# Copied from extract_simple_embeddings
def add_simple_embeddings(
    # self,
    target_df: pd.DataFrame,
    max_n_samples=None,
    skip_na_rows=False,
    inplace=True,
    add_system_prompt=False,
    embedding_model="word2vec",
    loaded_embedder=None,
    return_mean=True,
    return_std=False,
    normalise=False,
    filter_stopwords=False,
    overwrite_existing=False,
    verbose=False
):
    """
    Which embedding model should I use as default? fasttext or word2vec?
    word2vec is faster but cannot handle words that it has not seen during training, while fasttext can as it also
    considers sub-word information. However, fasttext is slower.

    :param max_n_samples: The maximum number of samples to extract embeddings for. If None, extract embeddings for
    all samples
    :param skip_na_rows: If True, discard all rows where there is at least one nan in the columns starting with
    "Success" for the considered llms
    :param inplace: If True, replace the original dataframe with the one with the embeddings. If False, return the
    new dataframe
    :param embedding_model: The embedding model to use, either "fasttext" or "word2vec". Default is "word2vec".
    :param loaded_embedder: If not None, the Embedder object to use for the embeddings. If None, a new Embedder
    object will be created.
    :param return_mean : bool, default=True
        Whether to return the mean of the embeddings of the words in the transcript or all embeddings.
    :param return_std : bool, default=True
        Whether to return the standard deviation of the embeddings of the words in the transcript, besides the mean
        or full embeddings.
    :param normalise : bool, default=False
        Whether to normalise the embeddings. If return_mean is True, this will normalise the mean vector. If
        return_mean is False, this will normalise each embedding.
    :param filter_stopwords : bool, default=False
        Whether to filter out stopwords.
    :param overwrite_existing : bool, default=False
        Whether to overwrite the existing embeddings if they exist.

    :return:
    """

    # self._check_system_prompt_exists(add_system_prompt)

    # prompts_df = self._extract_prompts_df(
    #     max_n_samples=max_n_samples,
    #     skip_na_rows=skip_na_rows,
    #     add_system_prompt=add_system_prompt,
    # )

    # check if the embeddings have already been computed; if not overwrite_existing, discard the rows where the
    # embeddings are already present
    # if not overwrite_existing:
    #     if f"{embedding_model}_embeddings" in self.results_df.columns:
    #         previous_length = len(prompts_df)
    #         prompts_df = prompts_df[
    #             ~prompts_df["prompt"].isin(
    #                 self.results_df[
    #                     self.results_df[f"{embedding_model}_embeddings"].notna()
    #                 ]["prompt"]
    #             )
    #         ]
    #         self._print_if_verbose(
    #             f"Discarding {previous_length - len(prompts_df)} rows as the embeddings have already been computed"
    #         )

    # if len(prompts_df) == 0:
    #     self._print_if_verbose("All embeddings have already been computed")
    #     if inplace:
    #         return self
    #     else:
    #         return self.results_df

    # if loaded_embedder is None:
        # self._print_if_verbose(
        #     f"Instantiating the Embedder with {embedding_model} model..."
        # )
    embedder = Embedder(
        embedding_model,
        return_mean=return_mean,
        return_std=return_std,
        normalise=normalise,
        verbose=verbose,
        filter_stopwords=filter_stopwords,
    )
    #     self._print_if_verbose(f"Done")
    # else:
    #     embedder = loaded_embedder
    #     embedder.update(
    #         return_mean=return_mean,
    #         return_std=return_std,
    #         normalise=normalise,
    #         verbose=self.verbose,
    #         filter_stopwords=filter_stopwords,
    #     )

    # self._print_if_verbose(f"Extracting embeddings for {max_n_samples} samples...")
    target_df[
        f"{embedding_model}_embeddings"
        # + ("_sys_prompt" if add_system_prompt else "")
    ] = target_df[
        "prompt"
        # + ("_sys_prompt" if add_system_prompt else "")
    ].apply(
        embedder
    )
    # self._print_if_verbose(f"Done")

    # now merge with the original df; if inplace, I will replace the original df
    # self._print_if_verbose("Merging with the original dataframe and return.")
    # if inplace:
    #     self.results_df = self.results_df.merge(prompts_df, on="prompt", how="left")
    #     return self
    # else:
    #     return self.results_df.merge(prompts_df, on="prompt", how="left")

def convert_schema_in_place(df: pd.DataFrame, llm_name) -> None:
    df["response"] = df["response"].astype(bool)
    rename_mapping = {
        "id": "question_id",
        "response": f"Success_{llm_name}"
    }

    df.drop(columns=["model", "timestamp", "target"], inplace=True)
    df.rename(columns=rename_mapping, inplace=True)

def load_open_llm_v2(llms: List[str], train_dataset_name: str, test_dataset_name: str, exclude_embeddings = False):
    columns = ["question_id", "prompt"]
    output_dfs = []
    for dataset_name, dataset_kind in [
            (train_dataset_name, 'train'),
            (train_dataset_name, 'val'),
            (test_dataset_name, 'test')
        ]:
        output_df = pd.DataFrame(columns=columns)
        # Add a success column for each LLM of interest
        for llm in llms:
            benchmark_data = process_benchmarks(llm, data_folder_location=DATA_FOLDER_LOCATION)
            try:
                llm_df = benchmark_data[dataset_name][dataset_kind]
            except KeyError as e:
                warn(f"Skipping LLM {llm} due to error: {e}")
                continue
            convert_schema_in_place(llm_df, llm)
            output_df = pd.merge(output_df, llm_df, on=columns, how="right")
        if not exclude_embeddings:
            # Add columns for emeddings
            for embedding_model in ["word2vec", "fasttext"]:
                print(f"Adding embeddings for {embedding_model}")
                add_simple_embeddings(output_df, embedding_model=embedding_model)
            # Add OpenAI embeddings
            print(f"Adding OpenAI embeddings")
            open_ai_embeddings_df = pd.read_json(MMLU_PRO_OPENAI_RESULTS_LOCATION)
            # Make sure question ID columns have the same type
            output_df['question_id'] = output_df['question_id'].astype(str)
            open_ai_embeddings_df['id'] = open_ai_embeddings_df['id'].astype(str)
            # Add OpenAI embedding for each prompt
            output_df = pd.merge(output_df, open_ai_embeddings_df, left_on="question_id", right_on="id", how="left")
            # Double check the id => prompt mapping is the same in both datasets (note the newlines are different so they need a bit of handling)
            assert (output_df['prompt_x'].str.replace(chr(13), '').replace(chr(10), '') == output_df['prompt_y'].str.replace(chr(13), '').replace(chr(10), '')).all(), "Mismatch between prompts in datasets"
            # Tidy up the duplicate prompt column now that we've checked it
            output_df = output_df.drop(columns=['prompt_y']).rename(columns={'prompt_x': 'prompt'})
        output_dfs.append(output_df)
    train_df, validation_df, test_df = output_dfs

    return train_df, validation_df, test_df