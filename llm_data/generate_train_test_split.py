import re

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from typing import List, Tuple, Dict
from warnings import warn

BENCHMARKS = ["Big-Bench-Hard", "IFEVAL", "MATH-LvL-5", "MMLU-Pro", "MuSR"]

SUBSETS = {"Big-Bench-Hard" : ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
                  "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
                  "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
                  "tracking_shuffled_objects_three_objects", "web_of_lies"], 
                  "IFEVAL" : ["train"], 
                  "MATH-LvL-5" : ["algebra_hard", "counting_and_prob_hard", "geometry_hard", "intermediate_algebra_hard", "num_theory_hard", "prealgebra_hard", "precalculus_hard"], 
                  "MMLU-Pro" : ["train"],
                  "MuSR" : ["musr_murder_mysteries", "musr_object_placements", "musr_team_allocation"]
                  }

SUBSETS = {"Big-Bench-Hard" : ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
                  "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
                  "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
                  "tracking_shuffled_objects_three_objects", "web_of_lies"], 
                  "IFEVAL" : ["IFEVAL"], 
                  "MATH-LvL-5" : ["algebra_hard", "counting_and_prob_hard", "geometry_hard", "intermediate_algebra_hard", "num_theory_hard", "prealgebra_hard", "precalculus_hard"], 
                  "MMLU-Pro" : ["MMLU-Pro"],
                  "MuSR" : ["musr_murder_mysteries", "musr_object_placements", "musr_team_allocation"]
                  }

def _load_prompts(bm: str) -> pd.DataFrame:
    """
    Loads in the prompts for a particular benchmark, concatenating all subsets. This is a helper function for `load_all_prompts`. We recommend using that instead.

    Args:
        bm (str): The name of the benchmark to be read in.
    
    Returns:
        pd.DataFrame: All prompts for that benchmark.
    """
    dataset = pd.read_parquet("hf://datasets/kvoudouris/PredictaBoard_Benchmarks/" + f"data/{SUBSETS[bm][0]}.parquet")
    if len(SUBSETS[bm]) > 1:
        all_prompts = []
        dataset['subset'] = [SUBSETS[bm][0]] * dataset.shape[0]
        all_prompts.append(dataset)
        for s in SUBSETS[bm][1:]:
            ds = pd.read_parquet("hf://datasets/kvoudouris/PredictaBoard_Benchmarks/" + f"data/{s}.parquet")
            ds['subset'] = [s] * ds.shape[0]
            all_prompts.append(ds)
        dataset = pd.concat(all_prompts, ignore_index=True)
        dataset['id'] = dataset['id'].astype(int)
    return dataset

def load_all_prompts(benchmarks: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads in the prompts for a list of benchmarks.

    Args:
        benchmarks (List[str]): The names of the benchmarks to be read in.
    
    Returns:
        Dict[pd.DataFrame]: A dictionary of all prompts for each benchmark.
    """
    assert all(b in BENCHMARKS for b in benchmarks), f"Invalid benchmark name. Choose from {BENCHMARKS}"
    datasets = {}
    for bm in benchmarks:
        print(f"Loading prompts for {bm}")
        datasets[bm] = _load_prompts(bm)
    return datasets

def _load_llm_data(llm: str, bm: str, timestamp: str = "latest") -> pd.DataFrame:
    """
    Loads in the results of an LLM with a particular timestamp on a particular benchmark. This is a helper function for `load_all_llm_data`. We recommend using that instead.

    Args:
        llm (str): The name of the LLM to be assessed.
        bm (str): The name of the benchmark to be read in.
        timestamp (str): The timestamp of the LLM to be read in. If "latest", the most recent timestamp is used.
    
    Returns:
        pd.DataFrame: A dataframe of all llm results for each benchmark.
    """
    def _filter_timestamp(ds):
        if timestamp == "latest":
            return ds[ds['timestamp'] == ds['timestamp'].max()]
        else:
            return ds[ds['timestamp'] == timestamp]
    assert timestamp == "latest" or bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}$", timestamp)), ""
    dataset = pd.read_parquet("hf://datasets/kvoudouris/PredictaBoard_LLMs/" + f"data/{SUBSETS[bm][0]}.parquet")
    dataset = dataset[dataset['model'] == llm]
    dataset = _filter_timestamp(dataset)
    if len(SUBSETS[bm]) > 1:
        all_results = []
        dataset['subset'] = [SUBSETS[bm][0]] * dataset.shape[0]
        all_results.append(dataset)
        for s in SUBSETS[bm][1:]:
            ds = pd.read_parquet("hf://datasets/kvoudouris/PredictaBoard_LLMs/" + f"data/{s}.parquet")
            ds = ds[ds['model'] == llm]
            ds = _filter_timestamp(ds)
            ds['subset'] = [s] * ds.shape[0]
            all_results.append(ds)
        dataset = pd.concat(all_results, ignore_index=True)
        dataset = pd.melt(dataset, id_vars=['model', 'timestamp', 'subset'], var_name='id', value_name='correct')
    else:
        dataset = pd.melt(dataset, id_vars=['model', 'timestamp'], var_name='id', value_name='correct')
    dataset['correct'] = pd.to_numeric(dataset['correct'], errors='coerce').fillna(-99).astype(int) # Handle any missing values
    dataset['id'] = pd.to_numeric(dataset['id'], errors='coerce').fillna(-99).astype(int) # Handle any missing values
    return dataset

def load_all_llm_data(llm_name: str, benchmarks: List[str], timestamp: str = "latest") -> Dict[str, pd.DataFrame]:
    """
    Loads in the results of an LLM with a particular timestamp on a list of benchmarks.

    Args:
        llm_name (str): The name of the LLM to be assessed.
        benchmarks (List[str]): The names of the benchmarks to be read in.
        timestamp (str): The timestamp of the LLM to be read in. If "latest", the most recent timestamp is used.
    
    Returns:
        Dict[pd.DataFrame]: A dictionary of all results for that LLM for each benchmark. Correct=1, Incorrect=0.
    """
    assert all(b in BENCHMARKS for b in benchmarks), f"Invalid benchmark name. Choose from {BENCHMARKS}"
    datasets = {}
    for bm in benchmarks:
        print(f"Loading LLM data for {bm}")
        datasets[bm] = _load_llm_data(llm_name, bm, timestamp)
    return datasets

def combine_and_split_data(all_prompts: Dict[str, pd.DataFrame], all_llm_data: Dict[str, pd.DataFrame], prop: float, val: bool = True, seed: int = 1234) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Combines the prompts and LLM data for each benchmark, and optionally splits the data into train, validation, and test sets.
    Set prop=0 to not split the data. Set val=False (default=True) to not include a validation set.

    Args:
        all_prompts (Dict[str, pd.DataFrame]): A dictionary of dataframes of prompts for each benchmark, produced by `load_all_prompts`.
        all_llm_data (Dict[str, pd.DataFrame]): A dictionary of dataframes of LLM results for each benchmark, produced by `load_all_llm_data`.
        prop (float): The proportion of data to be used for the test set.
        val (bool): Whether to include a validation set. Default is True.
        seed (int): The random seed to be used for the train-test split.
    
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: A dictionary of combined and splitted results. If prop=0, the key is "all". Otherwise, the keys are "train", "val" (if val=True), and "test".
    """
    assert sorted(all_prompts.keys()) == sorted(all_llm_data.keys()), "Mismatch in benchmarks between prompts and llm data."
    assert prop >= 0 and prop <= 1, "Proportion of data for test set must be between 0 and 1."
    combined_data = {}
    for bm in all_prompts.keys():
        prompts = all_prompts[bm]
        llm_data = all_llm_data[bm]
        if 'subset' in llm_data.columns:
            combined = pd.merge(prompts, llm_data, on=["id", "subset"], how="left")
        else:
            combined = pd.merge(prompts, llm_data, on="id", how="left")
        na_count = combined['correct'].isna().sum()
        if na_count > 0:
            print(f"Warning: {na_count} rows have missing values in the correct column. Ignoring for splits")
        if prop > 0:
            train_split, remainder = train_test_split(combined, test_size=prop, random_state=seed, shuffle = True)
            if val:
                print(f"Splitting into train ({1-prop}), val ({prop/2}), test ({prop/2})")
                val_split, test_split = train_test_split(remainder, test_size=0.5, random_state=seed, shuffle = True)
                combined_data[bm] = {"train": train_split, "val": val_split, "test": test_split}
            else:
                combined_data[bm] = {"train": train_split, "test": remainder}
        else:
            combined_data[bm] = {"all": combined}
    return combined_data

def example_loader() -> Dict[str, Dict[str, pd.DataFrame]]:
    all_prompts = load_all_prompts(["Big-Bench-Hard", "IFEVAL"])
    all_llm_data = load_all_llm_data("migtissera__Llama-3-70B-Synthia-v3.5", ["Big-Bench-Hard", "IFEVAL"])
    combined_data = combine_and_split_data(all_prompts, all_llm_data, 0.2, val=True)
    return combined_data
