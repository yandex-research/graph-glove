import os
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from lib import training_mode
from ...utils import download


class SimilarityDataset:

    def __init__(self, dataset, lowercase, data_path='./data', **kwargs):
        """
        Contains data required to evaluate similarity benchmarks
        :param dataset: a pre-defined dataset name (see SIMILARITY_BENCHMARKS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param lowercase: if True, calls str.lower on all word pairs
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param kwargs: if dataset is not in SIMILARITY_BENCHMARKS, provide two keys: word_pairs and scores
        """

        if dataset in SIMILARITY_BENCHMARKS:
            data_dict = SIMILARITY_BENCHMARKS[dataset](os.path.join(data_path, dataset), **kwargs)
        else:
            assert all(key in kwargs for key in ('word_pairs', 'scores')), \
                "Unknown dataset. Provide word_pairs and scores params"
            data_dict = kwargs

        self.data_path = data_path
        self.dataset = dataset

        self.word_pairs = data_dict['word_pairs']
        if lowercase:
            self.word_pairs = np.array([
                [pair[0].lower(), pair[1].lower()] for pair in self.word_pairs
            ])

        self.scores = data_dict['scores']


def fetch_WS353(path):
    """
    The WordSimilarity-353 Test Collection
    """
    data_path = os.path.join(path, 'combined.csv')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'wordsim353.zip')
        download("http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    data = pd.read_csv(data_path)

    return dict(
        word_pairs=data[['Word 1', 'Word 2']].values,
        scores=data['Human (mean)'].values
    )


def fetch_SCWS(path):
    """
    Stanford's Contextual Word Similarities
    """
    data_path = os.path.join(path, 'SCWS/ratings.txt')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'SCWS.zip')
        download("http://www-nlp.stanford.edu/~ehhuang/SCWS.zip", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    data = pd.read_csv(data_path, sep='\t', header=None, quoting=3)

    return dict(
        word_pairs=data[[1, 3]].values,
        scores=data[7].values
    )


def fetch_RW(path):
    """
    The Stanford Rare Word (RW) Similarity Dataset
    """
    data_path = os.path.join(path, 'rw/rw.txt')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'rw.zip')
        download("http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    data = pd.read_csv(data_path, sep='\t', header=None)

    return dict(
        word_pairs=data[[0, 1]].values,
        scores=data[2].values
    )


def fetch_SimLex(path):
    data_path = os.path.join(path, 'SimLex-999/SimLex-999.txt')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'SimLex-999.zip')
        download("https://fh295.github.io/SimLex-999.zip", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    data = pd.read_csv(data_path, sep='\t')

    return dict(
        word_pairs=data[['word1', 'word2']].values,
        scores=data['SimLex999'].values
    )


def fetch_SimVerb(path):
    data_path = os.path.join(path, 'data/SimVerb-3500.txt')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'simverb.zip')
        download("https://www.repository.cam.ac.uk/bitstream/handle/1810/264124/simverb-3500-data.zip?sequence=1&isAllowed=y", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    data = pd.read_csv(data_path, sep='\t', header=None)

    return dict(
        word_pairs=data[[0, 1]].values,
        scores=data[3].values
    )


SIMILARITY_BENCHMARKS = {
    'WS353': fetch_WS353,
    'SCWS': fetch_SCWS,
    'RW': fetch_RW,
    'SimLex': fetch_SimLex,
    'SimVerb': fetch_SimVerb,
}


@torch.no_grad()
def evaluate_similarity(model, *, lowercase, batch_size=256, datasets=tuple(SIMILARITY_BENCHMARKS.keys()), **kwargs):
    """ Evaluates word similarity benchmarks as spearman coefficient between model- and human-predicted similarities"""

    metrics = {}
    for dataset_name in datasets:
        sim_data = SimilarityDataset(dataset_name, lowercase=lowercase)
        word_occurences = Counter(word for word in sim_data.word_pairs.reshape(-1))
        lhs_words = set()  # words from which we compute paths, optimized for dijkstra
        for w1, w2 in sim_data.word_pairs:
            w1_idx = model.token_to_ix.get(w1)
            w2_idx = model.token_to_ix.get(w2)
            if w1_idx is not None and w2_idx is not None:
                lhs_words.add(max((w1, w2), key=word_occurences.get))
            elif w1_idx is not None:
                lhs_words.add(w1)
            elif w2_idx is not None:
                lhs_words.add(w2)

        lhs_words = sorted(lhs_words)

        with training_mode(model, is_train=False):
            dists = model.evaluate_distances_to_all_vertices(
                list(map(model.token_to_ix.get, lhs_words)),
                batch_size=batch_size, soft=False, **kwargs)  # [len(lhs_words), num_vertices]
        words_to_lhs_idx = {word: i for i, word in enumerate(lhs_words)}  # mapping from word to its index in dists

        similarities = []
        similarities_notna = []
        for w1, w2 in sim_data.word_pairs:
            w1_idx = model.token_to_ix.get(w1)
            w2_idx = model.token_to_ix.get(w2)

            if w1_idx is not None and w2_idx is not None:
                if w1 in lhs_words:
                    distance = dists[words_to_lhs_idx[w1]][w2_idx]
                else:  # w2 in lhs_words
                    distance = dists[words_to_lhs_idx[w2]][w1_idx]
            elif w1_idx is not None:
                distance = dists[words_to_lhs_idx[w1]].mean()
            elif w2_idx is not None:
                distance = dists[words_to_lhs_idx[w2]].mean()
            else:
                distance = float('inf')

            similarity = -distance

            similarities.append(similarity)

            if w1_idx is not None and w2_idx is not None:
                similarities_notna.append(similarity)
            else:
                similarities_notna.append(np.nan)

        metrics[dataset_name + '_infer_nan'] = spearmanr(similarities, sim_data.scores, nan_policy='raise').correlation
        metrics[dataset_name + '_omit_nan'] = spearmanr(
            similarities_notna, sim_data.scores, nan_policy='omit').correlation
    return metrics
