# list of models @ https://platform.openai.com/docs/models/continuous-model-upgrades
# cost @ https://openai.com/pricing and https://platform.openai.com/docs/deprecations/ for older models
import os

import pandas as pd

cost_per_token = {
    "gpt-4-0125-preview": 0.01e-3,
    "gpt-4-1106-preview": 0.01e-3,
    "gpt-4-0613": 0.03e-3,
    "gpt-4-32k-0613": 0.06e-3,
    "gpt-3.5-turbo-0125": 0.0005e-3,  # optimized for chat
    "gpt-3.5-turbo-1106": 0.001e-3,  # optimized for chat
    "gpt-3.5-turbo-instruct": 0.0015e-3,
    # instruct model, indicated as the replacement for old instructGPT models (similar capabilities to
    #  textx-davinci-003, compatible with legacy Completions endpoint)
    # older models, will be deprecated in June 24
    "gpt-4-0314": 0.03e-3,
    "gpt-4-32k-0314": 0.06e-3,
    "gpt-3.5-turbo-0613": 0.0015e-3,
    "gpt-3.5-turbo-16k-0613": 0.003e-3,
    "gpt-3.5-turbo-0301": 0.0015e-3,
    # even older models
    "text-ada-001": 0.0004e-3,
    "text-babbage-001": 0.0005e-3,
    "text-curie-001": 0.0020e-3,
    "text-davinci-001": 0.0200e-3,
    "text-davinci-002": 0.0200e-3,
    "text-davinci-003": 0.0200e-3,
    "ada": 0.0004e-3,
    "babbage": 0.0005e-3,
    "curie": 0.0020e-3,
    "davinci": 0.0200e-3,
}


# This dictionary is up-to-date as 2024-03-20


def compute_n_tokens(string, tokenizer):
    return len(tokenizer(string)["input_ids"])


def compute_cost(model_name, num_tokens):
    """
    Compute the cost of the model for a given number of tokens and turns.
    """
    return cost_per_token[model_name] * num_tokens


def load_with_conditions(filename, overwrite_res=False):
    if not os.path.exists(filename) or overwrite_res:
        print("File not found or overwrite requested. Creating new dataframe.")
        df = pd.DataFrame()
    elif filename.split(".")[-1] == "csv":
        print("Loading existing dataframe.")
        df = pd.read_csv(filename)
    elif filename.split(".")[-1] == "pkl":
        print("Loading existing dataframe.")
        df = pd.read_pickle(filename)
    else:
        raise ValueError("File format not recognized. Please use .csv or .pkl.")

    return df


def save_dataframe(filename, res_df):
    if filename.endswith(".csv"):
        res_df.to_csv(filename, index=False)
    elif filename.endswith(".pkl"):
        res_df.to_pickle(filename)
    else:
        raise ValueError("filename not recognized")


llms_dict = {
    "01-ai/yi-34b": "01-ai/yi-34b",
    "01-ai/yi-6b": "01-ai/yi-6b",
    "AlephAlpha/luminous-base": "AlephAlpha/luminous-base",
    "AlephAlpha/luminous-extended": "AlephAlpha/luminous-extended",
    "AlephAlpha/luminous-supreme": "AlephAlpha/luminous-supreme",
    "ai21/j2-grande": "ai21/j2-grande",
    "ai21/j2-jumbo": "ai21/j2-jumbo",
    "anthropic/claude-2.0": "anthropic/claude-2.0",
    "anthropic/claude-2.1": "anthropic/claude-2.1",
    "anthropic/claude-instant-1.2": "anthropic/claude-instant-1.2",
    "anthropic/claude-v1.3": "anthropic/claude-v1.3",
    "cohere/command": "cohere/command",
    "cohere/command-light": "cohere/command-light",
    "google/text-bison@001": "google/text-bison@001",
    "google/text-unicorn@001": "google/text-unicorn@001",
    "meta/llama-2-13b": "meta/llama-2-13b",
    "meta/llama-2-70b": "meta/llama-2-70b",
    "meta/llama-2-7b": "meta/llama-2-7b",
    "meta/llama-65b": "meta/llama-65b",
    "mistralai/mistral-7b-v0.1": "mistralai/mistral-7b-v0.1",
    "mistralai/mixtral-8x7b-32kseqlen": "mistralai/mixtral-8x7b-32kseqlen",
    "gpt-3.5-turbo-0613": "openai/gpt-3.5-turbo-0613",
    "gpt-4-0613": "openai/gpt-4-0613",
    "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
    "text-davinci-002": "openai/text-davinci-002",
    "text-davinci-003": "openai/text-davinci-003",
    "tiiuae/falcon-40b": "tiiuae/falcon-40b",
    "tiiuae/falcon-7b": "tiiuae/falcon-7b",
    "writer/palmyra-x-v2": "writer/palmyra-x-v2",
    "writer/palmyra-x-v3": "writer/palmyra-x-v3",
}
llms_helm = list(llms_dict.keys())

# keep two families of models (anthropic and meta's llama) in the test set
train_llms_helm = [
    "01-ai/yi-6b",
    "01-ai/yi-34b",
    "AlephAlpha/luminous-base",
    "AlephAlpha/luminous-supreme",
    "ai21/j2-grande",
    "ai21/j2-jumbo",
    "cohere/command",
    "google/text-bison@001",
    "google/text-unicorn@001",
    "mistralai/mixtral-8x7b-32kseqlen",
    "mistralai/mistral-7b-v0.1",
    "gpt-3.5-turbo-0613",
    "gpt-4-1106-preview",
    "text-davinci-002",
    "text-davinci-003",
    "tiiuae/falcon-7b",
    "writer/palmyra-x-v3",
    "writer/palmyra-x-v2",
]

validation_llms_helm = [
    "tiiuae/falcon-40b",
    "gpt-4-0613",
    "AlephAlpha/luminous-extended",
    "cohere/command-light",
]

test_llms_helm = [
    "anthropic/claude-2.1",
    "anthropic/claude-2.0",
    "anthropic/claude-instant-1.2",
    "anthropic/claude-v1.3",
    "meta/llama-2-70b",
    "meta/llama-2-13b",
    "meta/llama-2-7b",
    "meta/llama-65b",
]
