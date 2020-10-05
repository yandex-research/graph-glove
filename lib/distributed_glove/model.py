from enum import Enum

from torch import nn
import numpy as np

from lib import check_numpy
from .graph_embedding import ParallelGraphEmbedding


class GloveModel(nn.Module):
    class COOCMode(Enum):
        DOT_PRODUCT = 1
        DISTANCE = 2
        DISTANCE_SQUARED = 3

    def __init__(self, graph_embedding: ParallelGraphEmbedding, token_to_ix,
                 null_vertex=None, cooc_mode=COOCMode.DOT_PRODUCT, bias_initial_std=0.01):
        super().__init__()
        self.graph_embedding = graph_embedding
        num_embeddings = graph_embedding.num_vertices
        self.biases = nn.Embedding(num_embeddings, 1, sparse=graph_embedding.sparse)
        nn.init.normal_(self.biases.weight, std=bias_initial_std)
        self.token_to_ix = token_to_ix
        self.ix_to_token = {i: token for token, i in token_to_ix.items()}
        self.cooc_mode = cooc_mode
        self.null_vertex = null_vertex if null_vertex is not None else self.graph_embedding.num_vertices - 1

    def forward(self, *, vertex_norms_output, **paths_output):
        """ Predict log(co-occurences) given pre-computed paths. Only used with DistributedTrainer """

        paths_info = self.graph_embedding(**paths_output)
        distances = paths_info['target_distances']  # [batch_size, num_cols]
        row_bias = self.biases(paths_output['row_indices'])  # [batch_size, 1]
        col_bias = self.biases(paths_output['col_matrix'])[..., 0]  # [batch_size, num_cols]

        if self.cooc_mode == self.COOCMode.DISTANCE:
            prediction = row_bias + col_bias - distances
            mean_logp_paths = paths_info['logp_target_paths'].mean()
        elif self.cooc_mode == self.COOCMode.DISTANCE_SQUARED:
            prediction = row_bias + col_bias - distances ** 2
            mean_logp_paths = paths_info['logp_target_paths'].mean()
        elif self.cooc_mode == self.COOCMode.DOT_PRODUCT:
            norms_info = self.graph_embedding(**vertex_norms_output)
            vertex_norms = norms_info['target_distances'].view(self.graph_embedding.num_vertices)
            row_norms = vertex_norms[paths_output['row_indices'], None]  # [batch_size, 1]
            col_norms = vertex_norms[paths_output['col_matrix']]  # [batch_size, num_cols]
            dot_products = 0.5 * (row_norms ** 2 + col_norms ** 2 - distances ** 2)  # [batch_size, num_cols]
            mean_logp_paths = 0.5 * (paths_info['logp_target_paths'].mean() + norms_info['logp_target_paths'].mean())
            prediction = row_bias + col_bias + dot_products
        else:
            raise NotImplementedError("Unknown co-occurence prediction mode: {}".format(self.cooc_mode))

        return prediction, mean_logp_paths

    def evaluate_distances_to_all_vertices(self, source_ix, *, batch_size=None, dijkstra_parameters=None,
                                           callback=lambda x: x, **kwargs):
        assert np.ndim(source_ix) == 1, "source_ix must be a vector of vertex indices"
        batch_size = batch_size or len(source_ix)
        if dijkstra_parameters is None:
            dijkstra_parameters = self.graph_embedding.prepare_for_dijkstra()
        source_ix = np.array(source_ix, dtype=np.int32)
        targets = np.broadcast_to(np.arange(self.graph_embedding.num_vertices, dtype=np.int32),
                                  (len(source_ix), self.graph_embedding.num_vertices))

        dists = np.empty((len(source_ix), self.graph_embedding.num_vertices), dtype=np.float32)
        for batch_start in callback(range(0, len(source_ix), batch_size)):
            chunk_idx = slice(batch_start, batch_start + batch_size)
            paths = self.graph_embedding.compute_paths(
                source_ix[chunk_idx], targets[chunk_idx], dijkstra_parameters, **kwargs)
            dists[chunk_idx] = check_numpy(self.graph_embedding(**paths)['target_distances'])
        return dists

    def evaluate_norms(self, dijkstra_parameters=None, **kwargs):
        if dijkstra_parameters is None:
            dijkstra_parameters = self.graph_embedding.prepare_for_dijkstra()

        norms_info = self.graph_embedding.compute_paths(np.array([self.null_vertex], dtype='int32'),
                                                        np.arange(self.graph_embedding.num_vertices, dtype='int32')[None],
                                                        dijkstra_parameters, **kwargs)
        norms = check_numpy(self.graph_embedding(**norms_info)['target_distances'])
        return norms.reshape([self.graph_embedding.num_vertices])
