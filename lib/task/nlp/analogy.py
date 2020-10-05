import os
import zipfile
from itertools import product

import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from ...utils import download, training_mode


class AnalogyDataset:

    def __init__(self, dataset, *, lowercase, data_path='./data', **kwargs):
        """
        Contains data required to evaluate analogy benchmarks
        :param dataset: a pre-defined dataset name (see ANALOGY_DATASETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param lowercase: if True, calls str.lower on all word pairs
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param kwargs: if dataset is not in ANALOGY_DATASETS, provide two keys: word_pairs and scores
        """

        if dataset in ANALOGY_DATASETS:
            data_dict = ANALOGY_DATASETS[dataset](data_path)
        else:
            assert all(key in kwargs for key in ('example', 'question', 'answer')), \
                "Unknown dataset. Provide example, question and answer fields"
            data_dict = kwargs

        self.data_path = data_path
        self.dataset = dataset

        self.example = data_dict.get('example')
        self.question = data_dict.get('question')
        self.answer = data_dict.get('answer')

        if lowercase:
            self.example = np.array([[pair[0].lower(), pair[1].lower()] for pair in self.example])
            self.question = np.array([word.lower() for word in self.question])
            self.answer = np.array([word.lower() for word in self.answer])


def fetch_bats(path, category):
    if not os.path.exists(os.path.join(path, 'BATS_3.0')):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'BATS_3.0.zip')
        download("https://vecto-data.s3-us-west-1.amazonaws.com/BATS_3.0.zip", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)

    category_path = os.path.join(path, 'BATS_3.0', category)

    examples = []
    questions = []
    answers = []

    for file in os.listdir(category_path):
        pairs_for_type = []
        with open(os.path.join(category_path, file)) as f:
            for line in f:
                pairs_for_type.append(tuple(line.strip().split()))
        for pair_1, pair_2 in product(pairs_for_type, pairs_for_type):
            if pair_1 != pair_2:
                examples.append(pair_1)
                questions.append(pair_2[0])
                answers.append(pair_2[1])
    return dict(
        example=examples,
        question=questions,
        answer=answers,
    )


def fetch_google_analogy(path, size):
    path = os.path.join(path, 'google-analogy-test-set')
    dir_path = os.path.join(path, 'question-data')
    if not os.path.exists(dir_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'archive.zip')
        download("https://www.dropbox.com/s/5ck9egxeet5n992/google-analogy-test-set.zip?dl=1", archive_path)

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path)

    questions = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        with open(file_path) as f:
            if size == 'full':
                questions += f.readlines()
            elif size == 'sem':
                if 'gram' not in file_path:
                    questions += f.readlines()
            elif size == 'syn':
                if 'gram' in file_path:
                    questions += f.readlines()

    questions = np.array(list(map(str.split, questions)))

    return dict(
        example=questions[:, :2],
        question=questions[:, 2],
        answer=questions[:, 3]
    )


def fetch_MSR(path):
    data_path = os.path.join(path, 'MSR', 'EN-MSR.txt')
    if not os.path.exists(data_path):
        os.makedirs(os.path.join(path, 'MSR'), exist_ok=True)
        download("https://www.dropbox.com/s/gmlv57migftmj3x/EN-MSR.txt?dl=1", data_path)

    with open(data_path) as f:
        questions = f.readlines()

    questions = np.array(list(map(str.split, questions)))

    return dict(
        example=questions[:, :2],
        question=questions[:, 2],
        answer=questions[:, 4]
    )


ANALOGY_DATASETS = {
    'google_analogy_full': lambda path: fetch_google_analogy(path, size='full'),
    'google_analogy_sem': lambda path: fetch_google_analogy(path, size='sem'),
    'google_analogy_syn': lambda path: fetch_google_analogy(path, size='syn'),
    'msr_analogy': fetch_MSR,
    'bats_inf': lambda path: fetch_bats(path, '1_Inflectional_morphology'),
    'bats_der': lambda path: fetch_bats(path, '2_Derivational_morphology'),
    'bats_enc': lambda path: fetch_bats(path, '3_Encyclopedic_semantics'),
    'bats_lex': lambda path: fetch_bats(path, '4_Lexicographic_semantics'),
}


@torch.no_grad()
def evaluate_analogy(model, *, lowercase, batch_size=256, fast_rerank_k=10,
                     datasets=tuple(ANALOGY_DATASETS.keys()), eps=1e-9, callback=lambda x: x, **kwargs):
    """
    Evaluates the accuracy for predicting analogies. Uses tuples of (a, a_hat, b, b_hat)
    The task is to "find b_hat such that b_hat is to b same as a_hat is to a"

    :param lowercase: if True, all words in evaluation set will be lowercased
    :param batch_size: how many samples are processed at once when computing distances
    :param fast_rerank_k: pre-selects this many candidates for analogy using pseudo-cosine function (see below)
        Runtime complexity grows linearly in this parameter. After the candidates are selected, we can select
        the best candidate using 3COSADD(cand) = sim(cand, a_had) + sim(cand, b) - sim(cand, a)
        where sim(x, y) is pearson correlation between distances(x, all_vertices) and distances(y, all_vertices)
    :param datasets: which of ANALOGY_DATASETS to use for evaluation
    :param eps: small value used for numerical stability
    :param callback: optionally, tqdm
    :returns: a dictionary with analogy accuracies and non-OOV dataset sizes

    This pseudo-cosine function is based on how cosine works in euclidean space:
                                       (<b_hat, b> + <b_hat, a_hat> - <b_hat, a>)
    cosine(b_hat, b + a_hat - a) =   ----------------------------------------------
                                           ||b_hat|| * || b + a_hat - a ||

    Expanding dot product: <x, y> = 0.5 * (||x||^2 + ||y||^2 - ||x - y||^2)

                                     ||b_hat||^2 - ||b_hat - b||^2 - ||b_hat - a_hat||^2 + ||b_hat - a||^2 + const(b_hat)
    cosine(b_hat, b + a_hat - a) =   ------------------------------------------------------------------------------------
                                           ||b_hat|| * const(b_hat)


    Substituting graph-based distances: ||x - y|| = d_G(x, y) ; ||x|| = ||x - 0|| = d_G(x, v_0)
    where d_G(x, y) is a graph shortest path distance between x and y vertices, v_0 is "some zero vertex"

                                         d(b_hat, 0)^2 - d(b_hat, b)^2 - d(b_hat, a_hat)^2 + d(b_hat, a)^2
    pseudo_cosine(b_hat, b, a_hat, a) =  -----------------------------------------------------------------
                                                            d(b_hat, 0) + epsilon
    """

    word_dists_cache = {}
    dijkstra_parameters = model.graph_embedding.prepare_for_dijkstra()
    metrics = {}

    with training_mode(model, is_train=False):
        norms = model.evaluate_norms(soft=False, **kwargs)  # [num_vertices]

    for dataset_name in tqdm(datasets):
        data = AnalogyDataset(dataset_name, lowercase=lowercase)

        # 1. pre-compute distances from words in analogy dataset to all vertices
        words_to_process = []
        for (a, a_hat), b, b_hat in callback(zip(data.example, data.question, data.answer)):
            if all(word in model.token_to_ix for word in (a, a_hat, b, b_hat)):
                for word in (a, a_hat, b, b_hat):
                    if word not in word_dists_cache:
                        words_to_process.append(word)
        words_to_process = list(set(words_to_process))
        with training_mode(model, is_train=False):
            dists = model.evaluate_distances_to_all_vertices(
                list(map(model.token_to_ix.get, words_to_process)),
                soft=False, batch_size=batch_size, callback=callback,
                dijkstra_parameters=dijkstra_parameters, **kwargs
            )  # [len(lhs_words), num_vertices]
        for i, word in enumerate(words_to_process):
            word_dists_cache[word] = dists[i]

        # 2. find k most likely analogies based on pre-computed distances (with cheap proxy for analogy)

        tasks_with_candidates = []
        for (a, a_hat), b, b_hat in callback(zip(data.example, data.question, data.answer)):
            if all(word in model.token_to_ix for word in (a, a_hat, b, b_hat)):
                scores = (norms ** 2 - word_dists_cache[a_hat] ** 2 + word_dists_cache[a] ** 2
                          - word_dists_cache[b] ** 2) / (norms + eps)

                scores[[model.token_to_ix[a], model.token_to_ix[b], model.token_to_ix[a_hat], -1]] = -float('inf')
                top_cands = np.argpartition(scores, -fast_rerank_k)[-fast_rerank_k:]
                tasks_with_candidates.append(((a, a_hat, b, b_hat), top_cands))

        # 3. re-score k best candidates based on the graph 3COSADD (based on correlation between distances)

        unique_candidates = list(set(cand for _, cands in tasks_with_candidates
                                     for cand in cands if model.ix_to_token[cand] not in word_dists_cache))
        with training_mode(model, is_train=False):
            unique_candidates_dists = model.evaluate_distances_to_all_vertices(
                unique_candidates,
                soft=False, callback=callback, batch_size=batch_size,
                dijkstra_parameters=dijkstra_parameters, **kwargs
            )

        for i, cand in enumerate(unique_candidates):
            word_dists_cache[model.ix_to_token[cand]] = unique_candidates_dists[i]

        is_correct = []
        is_in_top = []
        for (a, a_hat, b, b_hat), cands in callback(tasks_with_candidates):
            cand_scores = []
            for cand in cands:
                candidate_dists_to_all = word_dists_cache[model.ix_to_token[cand]]
                similarity_cand_a = pearsonr(candidate_dists_to_all, word_dists_cache[a])[0]
                similarity_cand_b = pearsonr(candidate_dists_to_all, word_dists_cache[b])[0]
                similarity_cand_a_hat = pearsonr(candidate_dists_to_all, word_dists_cache[a_hat])[0]
                cand_scores.append(similarity_cand_a_hat - similarity_cand_a + similarity_cand_b)
            prediction = cands[np.argmax(cand_scores)]
            is_correct.append(model.token_to_ix[b_hat] == prediction)
            is_in_top.append(model.token_to_ix[b_hat] in cands)

        metrics[dataset_name + '_accuracy@1'] = np.mean(is_correct)
        metrics[dataset_name + f'_accuracy@{fast_rerank_k}'] = np.mean(is_in_top)
        metrics[dataset_name + '_n_samples'] = len(is_correct)
        print(dataset_name, np.mean(is_correct))
    return metrics
