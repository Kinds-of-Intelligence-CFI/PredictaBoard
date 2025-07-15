import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from time import sleep
from typing import List, Literal, Optional, Dict, Tuple
from logging import warning

import numpy as np
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tqdm import tqdm

from embeddings_utils.embedding import Embedder
from embeddings_utils.results_loaders import (
    ResultsLoader,
    _finalize_train_validation_test_dfs,
)

tqdm.pandas()


class MMLUProResultsLoader(ResultsLoader):
    BASE_RESULTS_PATH = os.path.join("download", "MMLU_Pro")

    def __init__(
        self,
        llms: List[str],
        base_path_raw=None,
        processed_filename=None,
        verbose=False,
        load_results_kwargs={},
        load_processed_kwargs={},
    ):
        super().__init__(
            llms=llms,
            base_path_raw=base_path_raw,
            processed_filename=processed_filename,
            verbose=verbose,
            load_results_kwargs=load_results_kwargs,
            load_processed_kwargs=load_processed_kwargs,
        )

    def load_results(self, base_path=None) -> pd.DataFrame:
        results_path = os.path.join(self.BASE_RESULTS_PATH, f"{self.llms[0]}.json")
        with open(results_path, "r") as file:
            llm_results_json = json.load(file)
        results = pd.DataFrame(
            [
                (elem["question_id"], elem["question"], elem["answer"] == elem["pred"])
                for elem in llm_results_json
            ],
            columns=["question_id", "prompt", f"Success_{self.llms[0]}"],
        )
        for llm in self.llms[1:]:
            results_path = os.path.join(self.BASE_RESULTS_PATH, f"{llm}.json")
            with open(results_path, "r") as file:
                llm_results_json = json.load(file)
            llm_results = pd.DataFrame(
                [
                    (
                        elem["question_id"],
                        elem["question"],
                        elem["answer"] == elem["pred"],
                    )
                    for elem in llm_results_json
                ],
                columns=["question_id", "prompt", f"Success_{llm}"],
            )
            question_ids_new = llm_results["question_id"].tolist()
            len_before = len(results)
            question_ids_before = results["question_id"].tolist()
            results = results.merge(
                llm_results, on=["question_id", "prompt"], how="inner"
            )
            if len(results) != len_before:
                question_ids_after = results["question_id"].tolist()
                non_intersection = set(question_ids_before) - set(question_ids_after)
                warning(
                    f"Adding LLM {llm} removed {len(non_intersection)} non intersecting question IDs: {non_intersection}"
                )
                unable_to_add_ids = set(question_ids_new) - set(question_ids_after)
                warning(
                    f"Unable to add {len(unable_to_add_ids)} IDs for LLM '{llm}' IDs: {unable_to_add_ids}"
                )
        return results


# TODO: merge into ResultsLoader
def load_mmlu_pro(
    llms: List[str],
    features: List[Literal["openai_embeddings", "word2vec", "fasttext"]],
    base_path="computed_embeddings/",
    validation_size=0.2,
    subsampled_n_train: Optional[int] = None,
    subsampled_n_test: Optional[int] = None,
    random_state=42,
):
    """

    :param llms: list of llms to consider
    :param features: list of features to consider (among "openai_embeddings", "word2vec", "fasttext")
    :param base_path:
    :param ood_split: If False, the train-test split will be done randomly;
    if OOD_1, the test split will be math, gsm, and mmlu abstract algebra;
    if OOD_2, the test split will be legalbench;
    if OOD_3, the test split will be commonsense, med_qa, and mmlu (except abstract algebra)
    :param validation_size: dedicate that fraction of the train split to validation. This is done after subsampling
    the training set (see below)
    :param subsampled_n_train: If not None, the train split will be subsampled to this number of rows
    :param subsampled_n_test: If not None, the test split will be subsampled to this number of rows
    :param random_state: random state for the subsampling

    :return:
    List of three dataframes: train_df, validation_df, test_df
    """
    openai_embeddings = "openai_embeddings" in features

    if openai_embeddings:
        raise NotImplementedError()

    # load with the embeddings:
    instance = MMLUProResultsLoader(
        llms=llms,
        processed_filename=f"{base_path}/mmlu_pro{'_with_embeddings.gz' if openai_embeddings else '.json'}",
        load_processed_kwargs=({"compression": True} if openai_embeddings else {}),
        verbose=True,
    )
    instance = instance.discard_if_one_na_in_success_per_row(
        inplace=True
    )  # this discards all rows where the considered llm has not been tested
    # Removed some system prompt stuff here TODO: Does MMLU Pro do any experiments with a system prompt?

    for embedding_type in ["word2vec", "fasttext"]:
        if embedding_type in features:
            # compute the word2vec and fasttext embeddings
            instance.extract_simple_embeddings(
                skip_na_rows=True,
                embedding_model=embedding_type,
                filter_stopwords=True,
                add_system_prompt=False,
            )

        # Removed some HELM stuff re OOD split

    # Using the built in train test split as (1) we don't do any pooling as in helm (2) we need to make use of getting rid of the NaN rows

    train_df, test_df = instance.train_test_split(
        0.7, rng=random_state, discard_na_rows=True
    )

    train_df, validation_df, test_df = _finalize_train_validation_test_dfs(
        train_df,
        test_df,
        validation_size,
        subsampled_n_train,
        subsampled_n_test,
        random_state,
    )

    return train_df, validation_df, test_df
