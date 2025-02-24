# Datasets

In this folder we include the datasets of LLM responses to benchmark prompts.

Currently, we have results from LLMs on the `open-llm-leaderboard-v2`. To add more LLMs from this repository, do the following.

1. Use `rye` (or a similar toml-based package manager) to install the required libraries.
2. Create a huggingface API token so that you can download data from huggingface.
3. Run the following code to download leaderboard results, changing the path in the curly braces.
```console
huggingface-cli download --resume-download --repo-type dataset open-llm-leaderboard/results --local-dir {path/to/save/hf-files} --local-dir-use-symlinks False
```
4. Run from this repo as root `rye run python src/predbench/synthesise_leaderboardv2_data.py --datadir {path/to/save/hf-files} --outputdir ./data/open-llm-leaderboard-v2 -r`.
5. Note that you will probably have to gain access to each repository individually to scrape the results. 