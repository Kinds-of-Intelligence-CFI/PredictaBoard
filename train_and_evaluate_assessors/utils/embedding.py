import os
from time import sleep

import gensim.downloader
import nltk
import numpy as np
import openai
from tqdm import tqdm


def get_openai_embedding(texts, model, sleep_time):
    # texts here is a list
    texts = [text.replace("\n", " ") for text in texts]
    while True:
        try:
            outs = openai.Embedding.create(input=texts, model=model)
            return [np.array(out["embedding"]) for out in outs["data"]]
        except:
            sleep(sleep_time)


def extract_openai_embeddings(prompts_df, embedding_model, batch_size, sleep_time_if_fail, col_name="prompt"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    embeddings_list = []
    for i in tqdm(range(0, len(prompts_df), batch_size)):
        # extract the prompts in the batch
        batch_prompts = (
            prompts_df[col_name]
            .iloc[i: i + batch_size]
            .tolist()
        )
        # extract the embeddings for the batch
        embeddings = get_openai_embedding(
            batch_prompts, embedding_model, sleep_time_if_fail
        )
        embeddings_list.extend(embeddings)
    return embeddings_list


# Adapted from the lie detection code: https://github.com/lorypack/llm-liedetector
class Embedder:
    def __init__(
            self,
            model,
            return_mean=True,
            return_std=False,
            normalise=False,
            verbose=False,
            filter_stopwords=False,
    ):
        """Embeds a transcript into a vector.

        Parameters
        ----------
        model : str
            The embeddings to use. Can be "fasttext" or "word2vec".
        return_mean : bool, default=True
            Whether to return the mean of the embeddings of the words in the transcript or all embeddings.
        return_std : bool, default=True
            Whether to return the standard deviation of the embeddings of the words in the transcript, besides the mean
            or full embeddings.
        normalise : bool, default=False
            Whether to normalise the embeddings. If return_mean is True, this will normalise the mean vector. If
            return_mean is False, this will normalise each embedding.
        verbose : bool, default=False
            Whether to print the unknown words while creating the embeddings.
        filter_stopwords : bool, default=False
            Whether to filter out stopwords.
        """

        if verbose:
            print("Loading embeddings...")
        if model == "fasttext":
            self.model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
        elif model == "word2vec":
            self.model = gensim.downloader.load("word2vec-google-news-300")
        if verbose:
            print("Embeddings loaded")

        self.return_mean = return_mean
        self.return_std = return_std
        self.normalise = normalise
        self.number_unknown_words = 0
        self.verbose = verbose
        if filter_stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))
        else:
            self.stopwords = set()

    def update(
            self,
            return_mean=None,
            return_std=None,
            normalise=None,
            verbose=None,
            filter_stopwords=None,
    ):
        if return_mean is not None:
            self.return_mean = return_mean
        if return_std is not None:
            self.return_std = return_std
        if normalise is not None:
            self.normalise = normalise
        if verbose is not None:
            self.verbose = verbose
        if filter_stopwords is not None:
            if filter_stopwords:
                self.stopwords = set(nltk.corpus.stopwords.words("english"))
            else:
                self.stopwords = set()

    def state(self):
        return {
            "model": self.model,
            "return_mean": self.return_mean,
            "return_std": self.return_std,
            "normalise": self.normalise,
            "verbose": self.verbose,
        }

    def __call__(self, sentence):
        """

        :param sentence: Needs to be a single string

        :return:

        """
        tokens = nltk.word_tokenize(sentence)

        vectors = []
        for word in tokens:
            if word not in self.stopwords:
                try:
                    vectors.append(list(self.model[word]))
                except KeyError:
                    self.number_unknown_words += 1
                    if self.verbose:
                        print(f"Word {word} not in vocabulary")
        if len(vectors) == 0:
            return list(np.zeros(self.model.vector_size))
        vectors = np.array(vectors)

        if self.return_mean:
            return_list = [vectors.mean(axis=0)]
        else:
            return_list = [vectors]
        if self.return_std:
            return_list.append(vectors.std(axis=0))

        if self.normalise:
            for i in range(len(return_list)):
                if return_list[i].ndim > 1:
                    return_list[i] = (
                            return_list[i] / np.linalg.norm(return_list[i], axis=1)[:, None]
                    )
                else:
                    return_list[i] = return_list[i] / np.linalg.norm(return_list[i])

        return (
            return_list[0]
            if not self.return_std
            else np.concatenate(return_list, axis=0)
        )

    def zero_pad(self, embeddings_list, max_length=None):
        if max_length is None:
            max_length = max([x.shape[0] for x in embeddings_list])

        matrix = np.zeros(
            (len(embeddings_list), max_length, embeddings_list[0].shape[1])
        )
        for i, sentence in enumerate(embeddings_list):
            x = np.array(sentence)
            matrix[i, : x.shape[0], :] = x

        return matrix


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_json("data/wikidata_refined.json")
    data = data[data.lie_transcript.notnull()]
    transcript = data.lie_transcript.values[0]

    sentence = "The quick brown fox jumps over the lazy dog."
    embedder = Embedder("fasttext", return_mean=True, normalise=True)
    vector = embedder(transcript)
    print(vector)
    print(np.linalg.norm(vector))

    embedder.return_mean = False
    vector = embedder(transcript)
    print(vector)
    print(vector.shape)
    print(np.linalg.norm(vector, axis=1))

    transcripts = data.lie_transcript.values[:10]
    embedding_list = [embedder(t) for t in transcripts]
    print(max([x.shape[0] for x in embedding_list]))
    zero_padded = embedder.zero_pad(embedding_list)
    print(zero_padded.shape)
