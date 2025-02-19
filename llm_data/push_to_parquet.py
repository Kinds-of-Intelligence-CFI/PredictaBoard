import argparse
import glob
import os

from pandas import read_csv

BM_SUBSETS = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
             "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
             "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
             "tracking_shuffled_objects_three_objects", "web_of_lies", "IFEVAL", "beginner_algebra_hard", "counting_and_prob_hard", "geometry_hard", "intermediate_algebra_hard", "num_theory_hard", "prealgebra_hard", 
             "precalculus_hard", "MMLU-Pro", "murder_mysteries", "object_placements", "team_allocation"]

def push_benchmark_prompts_to_parquet(directory: str, save_path: str) -> None:
    """
    Pushes local benchmark prompts to local clone of HF repository. Intended only for use by repository administrators.
    The prompt datasets have been scraped independently and stored as CSVs.
    """
    assert os.path.isdir(directory), f"Directory {directory} does not exist"
    files = glob.glob(f'{directory}/*_prompts.csv')

    assert len(files) == len(BM_SUBSETS), "Number of prompt files does not match number of benchmark subsets"

    for f in files:
        dataset = read_csv(f)
        subset = [s for s in BM_SUBSETS if s in f][0]
        dataset.to_parquet(os.path.join(save_path, "PredictaBoard_Benchmarks", "data", f"{subset}.parquet"))

def push_llm_performance_results_to_parquet(directory: str, save_path: str) -> None:
    """
    Pushes LLM performance results to local clone of HF repository. Intended only for use by repository administrators.
    The prompt datasets have been scraped independently and stored as CSVs.
    """
    assert os.path.isdir(directory), f"Directory {directory} does not exist"
    files = glob.glob(f'{directory}/*_results.csv')

    assert len(files) == len(BM_SUBSETS), "Number of prompt files does not match number of benchmark subsets"
    
    files.sort()

    for f in files:
        dataset = read_csv(f)
        subset = [s for s in BM_SUBSETS if s in f][0]
        dataset.to_parquet(os.path.join(save_path, "PredictaBoard_LLMs", "data", f"{subset}.parquet"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push benchmark prompts to huggingface repository")
    parser.add_argument("--prompt_directory", type=str, help="Directory containing benchmark prompt CSVs", default=None)
    parser.add_argument("--llm_directory", type=str, help="Directory containing LLM performance", default=None)
    parser.add_argument("--parquet_file_loc", type=str, help="Location to save parquet files. This directory must contain the PredictaBoard_Benchmarks and PredictaBoard_LLMs HF repositories.")
    args = parser.parse_args()
    if args.prompt_directory:
        print("Pushing benchmark prompts to parquet...")
        push_benchmark_prompts_to_parquet(args.prompt_directory, args.parquet_file_loc)
    if args.llm_directory:
        print("Pushing LLM performance results to parquet")
        push_llm_performance_results_to_parquet(args.llm_directory, args.parquet_file_loc)

