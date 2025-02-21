import argparse
import csv
import os
import re

import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import list_repo_files

SUBSETS = {"Big-Bench-Hard" : ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
                  "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
                  "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
                  "tracking_shuffled_objects_three_objects", "web_of_lies"], 
                  "IFEVAL" : ["IFEVAL"], 
                  "MATH-LvL-5" : ["beginner_algebra_hard", "counting_and_prob_hard", "geometry_hard", "intermediate_algebra_hard", "num_theory_hard", "prealgebra_hard", "precalculus_hard"], 
                  "MMLU-Pro" : ["MMLU-Pro"],
                  "MuSR" : ["murder_mysteries", "object_placements", "team_allocation"]
                  }

SUBSET_HF_ABBRVS = {"Big-Bench-Hard" : "bbh",
                  "IFEVAL" : "ifeval",
                  "MATH-LvL-5" : "math",
                  "MMLU-Pro" : "mmlu_pro",
                  "MuSR" : "musr"}

if os.path.isdir("./llm_performance_data"):
    CSVS_PATH = "./llm_performance_data"
elif os.path.isdir("./llm_data/llm_performance_data"):
    CSVS_PATH = "./llm_data/llm_performance_data"
else:
    raise FileExistsError("Cannot find LLM performance data CSV files. They should be in PredictaBoard/llm_data/llm_performance_data/")


def _check_model_store(model: str, stamp: str, path: str):
    file_exists = os.path.isfile(path)
    if file_exists:
        with open(path) as csvfile:
            csv_read = csv.reader(csvfile)
            for row in csv_read:
                if row[0] == model and row[1] == stamp:
                    return True
                else:
                    pass
            return False
    else:
        raise FileExistsError(f"Cannot find LLM performance data file at {path}")

def load_hf_instance_results(llm_name: str) -> None:
    """
    Updates results CSVs with a single LLM from the leaderboard. Assumes you are logged in to huggingface locally and you have access to the repository in question. It also assumes you are running from this folder or the root of the repo.

    Args:
        llm_name (str): The name of the llm in question. You can find a list in the details collection here https://huggingface.co/open-llm-leaderboard. Expects names in the form {hf_username}__{model_name} or {hf_username}/{model_name}
    
    Returns:
        Nothing. The CSVs in llm_performance_data are updated.
    """
    llm_name = llm_name.replace("open-llm-leaderboard", "").replace("-details", "").replace("/", "__") # handle miswritten names
    assert re.fullmatch(r'[^\_]+__[^\_]+', llm_name), f"Invalid LLM name {llm_name}. It must be in the format `hf-username`__`model-name` or `hf-username`/`model-name`. Edit this assertion if this regex wrongfully excludes a valid model."
    print(f"Attempting to scrape instance results for {llm_name}...")
    hf_files = [x for x in list_repo_files(f"open-llm-leaderboard/{llm_name}-details", repo_type="dataset") if "samples_leaderboard_" in x]
    timestamps = set([str(re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6})', t).group(1)) for t in hf_files])
    print(f"Found {len(timestamps)} timestamp(s).")
    for t in timestamps:
        timestamp_string = t
        t = "latest" if t == max(timestamps) else t.replace("-", "_") # deal with "latest" timestamp
        timestamp_legal = True # Some timestamps for some repositories do not have an associated split
        for bm in SUBSETS:
            if timestamp_legal:
                subsets = SUBSETS[bm]
                for s in subsets:
                    csv_path = f"{CSVS_PATH}/{bm}__{s}_results.csv" if len(subsets) > 1 else f"{CSVS_PATH}/{s}_results.csv"
                    check = _check_model_store(llm_name, timestamp_string, csv_path)
                    stored_data = pd.read_csv(csv_path)
                    if not check:
                        subset_name = s.replace("beginner_", "") #remove the 'beginner' that PredictaBoard uses to distinguish math subsets
                        bm_abbrv = SUBSET_HF_ABBRVS[bm]
                        try:
                            if len(subsets) > 1:
                                data = load_dataset(f"open-llm-leaderboard/{llm_name}-details", name=f"{llm_name}__leaderboard_{bm_abbrv}_{subset_name}", split=t)
                            else:
                                data = load_dataset(f"open-llm-leaderboard/{llm_name}-details", name=f"{llm_name}__leaderboard_{bm_abbrv}", split=t)
                        except ValueError:
                            timestamp_legal = False
                            print(f"Split not found for timestamp {timestamp_string}")
                            break
                        except DatasetNotFoundError as e:
                            print(f"{e}")
                            print("Make sure the repository is spelt correctly.")
                            print(f"If it is, make sure you have access. Visit https://huggingface.co/datasets/open-llm-leaderboard/{llm_name}-details to ask for access.")
                        col_names = ['model', 'timestamp'] + [str(c) for c in data['doc_id']]
                        if bm == "IFEVAL":
                            llm_row = [int(i) for i in data['prompt_level_strict_acc']]
                        elif bm == "MATH-LvL-5":
                            llm_row = data['exact_match']
                        elif bm == "MMLU-Pro":
                            llm_row = [int(i) for i in data['acc']]
                        else:
                            llm_row = [int(i) for i in data['acc_norm']]
                        llm_row = [llm_name, timestamp_string] + llm_row
                        new_row = pd.DataFrame([llm_row], columns = col_names)
                        updated_df = pd.concat([stored_data, new_row], ignore_index=True, axis=0)
                        updated_df.to_csv(csv_path, index=False)
                    else:
                        print(f"{llm_name} for timestamp {t} detected in {csv_path}. Moving on.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--llm_name', type=str, required=True)
    args = parser.parse_args()

    load_hf_instance_results(args.llm_name)
    print("We recommend clearing out your huggingface cache if you update with a lot of models. They can use quite a lot of memory.")