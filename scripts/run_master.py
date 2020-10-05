"""
Run master training process that handles GraphGloVe training. Requires at least one worker (see example in README.md)
"""
import argparse
import os
import sys

import numpy as np
import torch
from glove.corpus import Corpus

# setup path to allow importing lib
sys.path.insert(0, os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2]))

from lib.distributed_glove import DistributedGloveDataset, make_graph_from_vectors, GloveModel
from lib.distributed_glove.graph_embedding import ParallelGraphEmbedding
from lib.distributed_glove.trainer import DistributedTrainer
from lib.utils.job_server import JobServer


def run_master(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # start server (or connect to existing one)
    job_server = JobServer(host=args.host, port=args.port, password=args.password)

    # verify benchmarks
    similarity_benchmarks = args.track_benchmarks.split(',')
    for benchmark_name in similarity_benchmarks:
        assert benchmark_name in SIMILARITY_BENCHMARKS, "{} not found in DATASETS[{}]".format(
            benchmark_name, ', '.join(SIMILARITY_BENCHMARKS.keys()))
    assert hasattr(GloveModel.COOCMode, args.cooc_prediction_mode.upper()), \
        "unknown prediction mode {}".format(args.cooc_prediction_mode)

    # load training data
    corpora = Corpus.load(os.path.join(args.data_dir, args.cooc_name))
    corpora.matrix = corpora.matrix.astype(np.float32)

    if args.lowercase:
        for word in corpora.dictionary:
            assert word.islower() or not word.isalpha(), f"Training in lowercase mode but found uppercase token {word}." \
                                                         f"If you intend to use non-lowercased data, please add --non-lowercase to run_master.py call"
    dataset = DistributedGloveDataset(corpora=corpora, job_server=job_server, lowercase=args.lowercase)

    # initialize graph
    model_init_path = os.path.join(args.data_dir, f'init_prodige_{args.init}'
                                                  f'_soft{args.soft}'
                                                  f'_sparse{args.sparse}'
                                                  f'_init{args.init}'
                                                  f'_knn{args.knn_edges}'
                                                  f'_random{args.random_edges}'
                                                  f'_seed{args.seed}'
                                                  f'.pth')

    if not os.path.exists(model_init_path) or args.force_create_graph:
        glove_init_npz = np.load(os.path.join(args.data_dir, args.glove_init_name))

        glove_vectors = glove_init_npz['vectors']
        glove_biases = glove_init_npz['biases']
        glove_vectors = glove_vectors / np.linalg.norm(glove_vectors, axis=-1, keepdims=True)

        graph_embedding = make_graph_from_vectors(
            np.concatenate([glove_vectors, np.zeros_like(glove_vectors[:1])], axis=0),
            knn_edges=args.knn_edges, random_edges=args.random_edges, directed=False, soft=args.soft,
            max_length=args.max_length, n_jobs=args.n_jobs, k_nearest=0, sparse=args.sparse, verbose=True,
            GraphEmbeddingClass=ParallelGraphEmbedding,
        )
        model = GloveModel(
            graph_embedding, token_to_ix=corpora.dictionary, null_vertex=graph_embedding.num_vertices - 1,
            cooc_mode=getattr(GloveModel.COOCMode, args.cooc_prediction_mode))
        with torch.no_grad():
            model.biases.weight.view(-1).data[:len(glove_biases)] = torch.as_tensor(glove_biases)
        print(f"Caching initialized graph to {model_init_path}")
        torch.save(model, model_init_path)
    else:
        model = torch.load(model_init_path)

    # create trainer and fit
    pge_trainer = DistributedTrainer(
        model, dataset, job_server, experiment_name=args.exp_name, warm_start=True,
        base_lambda=args.base_lambda, lambda_warmup=args.lambda_warmup, clip_prob_max=args.clip_prob_max,
        report_frequency=args.report_frequency, upload_frequency=args.upload_frequency,
        optimizer_kwargs=dict(lr=args.lr), keep_checkpoints=args.keep_checkpoints,
        similarity_benchmarks=similarity_benchmarks, verbose=True)

    job_server.update_version(by=1, await_seconds=5)
    job_server.reset_queue()
    if args.evaluate_at_init:
        pge_trainer.evaluate_metrics()
    pge_trainer.fit(batch_size=args.batch_size, buffer_size_multiple=args.buffer_size_multiple)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--not_lowercase', action='store_false', dest='lowercase')
    parser.add_argument('--cooc_prediction_mode', default='DISTANCE', type=str)
    parser.add_argument('--report_frequency', type=int, default=50)
    parser.add_argument('--upload_frequency', type=int, default=5)
    parser.add_argument('--buffer_size_multiple', type=int, required=True)
    parser.add_argument('--data_dir', default='word_embeddings_50k', type=str)
    parser.add_argument('--cooc_name', default='cooc.pkl', type=str)
    parser.add_argument('--glove_init_name', default='init_glove.npz', type=str)
    parser.add_argument('--track_benchmarks', default='SCWS,WS353', type=str)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--base_lambda', default=10.0, type=float)
    parser.add_argument('--lambda_warmup', default=1000, type=int)
    parser.add_argument('--clip_prob_max', default=None, type=float)
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--knn_edges', type=int, default=64)
    parser.add_argument('--random_edges', type=int, default=10)
    parser.add_argument('--force_create_graph', action='store_true')
    parser.add_argument('--keep_checkpoints', type=int, default=10)
    parser.add_argument('--evaluate_at_init', action='store_true')
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1337)
    parsed_args = parser.parse_args()
    run_master(parsed_args)
