# LLM Data

This directory contains the results of several Large Language Models on the following benchmarks:
* Big-Bench-Hard
* IFEVAL
* MATH-LvL-5
* MMLU-Pro
* MuSR

The code in this directory was developed in a conda environment using the specifications in `environment.yml`.

## Loading Benchmark Prompts

To load the benchmark data in the required format, follow the instructions [here](https://huggingface.co/datasets/kvoudouris/PredictaBoard_Benchmarks), under `Use this dataset`.

## Loading LLM Responses

To load LLM responses in the required format, follow the instructions [here](https://huggingface.co/datasets/kvoudouris/PredictaBoard_LLMs), under `Use this dataset`.

To produce train, validation, and test splits for evaluating assessors, use `generate_train_test_split.py`.

## Contributing New LLM Responses

To contribute new LLM responses from the Open-LLM-Leaderboard-v2, do the following:
1. Identify the LLM entrant on huggingface that you wish to contribute.
2. Ensure that you are logged in to huggingface on your local machine and that you have requested access to the relevant repository.
3. Use `contribute_leaderboard_llm.py` to update the CSV files.
4. Open a pull request to ingest the new data. We will then review and push the new file to huggingface.

To contribute new LLM responses from models not involved in the leaderboard, things are a little more involved. We do not provide explicit support here. Here are some suggestions:
* Ideally, run evaluation using EleutherAI's `lm-evaluation-harness` for all benchmarks, using the following command line prompt `lm-eval...`. Process the data and append them to the relevant CSV files, and then make a PR.
* The harness requires access to logits for some benchmarks, meaning that many closed-source models cannot be evaluated on them. In this case, you will have to develop your own pipeline for evaluating models on these. We would be happy to discuss this further as we would like to include as many models as possible. Feel free to open a discussion on the repository.