import numpy as np
import scipy.sparse
import torch
from torch.nn import functional as F

from lib import GraphEmbedding
from lib.cpp import batch_dijkstra
from lib.task.nlp import make_graph_from_vectors


class ParallelGraphEmbedding(GraphEmbedding):
    """ A wrapper for GraphEmbedding that supports distributed_glove distributed_glove """

    def prepare_for_dijkstra(self):
        # make sure padding edge has weight 0 and probability 1
        with torch.no_grad():
            self.edge_adjacency_logits.data[:1].fill_(self.INF)
            self.edge_weight_logits.data[:1].fill_(self.NEG_INF)

        if self.directed:
            slices, edge_indices = self.slices, self.edge_targets
            edge_adjacency_logits = self.edge_adjacency_logits
            edge_weight_logits = self.edge_weight_logits
        else:
            slices, edge_indices = self.directed_slices, self.directed_edge_indices
            edge_adjacency_logits = F.embedding(
                    torch.as_tensor(self.reorder_undirected_to_directed),
                    self.edge_adjacency_logits, sparse=self.sparse
            )  # [num_edges, 1]
            edge_weight_logits = F.embedding(
                    torch.as_tensor(self.reorder_undirected_to_directed),
                    self.edge_weight_logits, sparse=self.sparse
            )  # [num_edges, 1]

        edge_logits = edge_adjacency_logits.data.numpy().flatten()

        if self.training:
            min_value, max_value = np.finfo(edge_logits.dtype).min, np.finfo(edge_logits.dtype).max
            edge_exists = (torch.rand(len(edge_logits)) < torch.sigmoid(torch.as_tensor(edge_logits))).numpy()
            edge_logits = np.where(edge_exists, max_value, min_value)

        return dict(
                slices=slices,
                sliced_edges=edge_indices,
                sliced_adjacency_logits=edge_logits,
                sliced_weight_logits=edge_weight_logits.data.numpy().flatten(),
        )

    def compute_paths(self, row_indices, col_matrix, dijkstra_parameters, **task_parameters):
        # set defaults
        parameters_modif = dict(self.defaults, **task_parameters)
        assert parameters_modif.get('max_length') is not None, "Please specify max_length in either init or forward"
        assert not parameters_modif.get('presample_edges', False)
        parameters_modif['deterministic'] = parameters_modif.get('deterministic', not self.training)
        parameters_modif['k_nearest'] = parameters_modif.get('k_nearest', 0)

        row_indices_tensor = torch.from_numpy(row_indices).to(dtype=torch.int32)
        col_matrix_tensor = torch.from_numpy(col_matrix).to(dtype=torch.int32)

        assert row_indices_tensor.device == col_matrix_tensor.device == torch.device('cpu'), "gpu not supported (yet)"

        target_paths, nearest_paths = batch_dijkstra(
                initial_vertices=row_indices_tensor.data.numpy(),
                target_vertices=col_matrix_tensor.data.numpy(),
                **dijkstra_parameters,
                **parameters_modif
        )
        return dict(
                row_indices=row_indices,
                col_matrix=col_matrix,
                target_paths=target_paths,
                nearest_paths=nearest_paths,
                **parameters_modif,
        )

    def _get_logits(self, sliced_logits, sliced_indices):
        """ A private helper function that returns logits of corresponding to indices with (maybe) sparse grad """
        if not self.directed:
            sliced_indices = self.reorder_undirected_to_directed[sliced_indices]
        if self.sparse:
            return F.embedding(sliced_indices, sliced_logits, sparse=True).view(*sliced_indices.shape)
        return sliced_logits[sliced_indices].view(*sliced_indices.shape)

    def _get_default_distance(self):
        """ A special magic that returns default distance in a way that gradients wrt that distance will be sparse"""
        if self.sparse:
            return F.embedding(torch.zeros(1, dtype=torch.int64, device=self.default_distance.device),
                               self.default_distance, sparse=True).view([])
        return self.default_distance.view([])

    def forward(self, row_indices, col_matrix, **parameters):
        # set defaults
        parameters = dict(self.defaults, **parameters)
        assert parameters.get('max_length') is not None, "Please specify max_length in either init or forward"
        parameters['deterministic'] = parameters.get('deterministic', not self.training)
        parameters['k_nearest'] = parameters.get('k_nearest', 0)

        assert all(
                param in parameters for param in ('target_paths', 'nearest_paths')), "Please call compute_paths first"
        device = self.edge_weight_logits.device
        target_paths = torch.as_tensor(parameters['target_paths'], device=device)
        nearest_paths = torch.as_tensor(parameters['nearest_paths'], device=device)

        # make sure padding edge has weight 0 and probability 1
        with torch.no_grad():
            self.edge_adjacency_logits.data[:1].fill_(self.INF)
            self.edge_weight_logits.data[:1].fill_(self.NEG_INF)

        # assert row_indices.device == col_matrix.device == torch.device('cpu'), "gpu not supported (yet)"

        row_indices = torch.as_tensor(row_indices, dtype=torch.int32, device=device)
        col_matrix = torch.as_tensor(col_matrix, dtype=torch.int32, device=device)
        if self.directed:
            slices, edge_indices = self.slices, self.edge_targets
        else:
            slices, edge_indices = self.directed_slices, self.directed_edge_indices

        target_paths = torch.as_tensor(target_paths, dtype=torch.int64, device=device)  # [batch_size, max_length]
        target_distances = F.softplus(self._get_logits(self.edge_weight_logits, target_paths)).sum(dim=(-1))
        logp_target_paths = -F.softplus(-self._get_logits(self.edge_adjacency_logits, target_paths)).sum(dim=(-1))
        # ^--[batch_size, num_targets]

        # handle paths that are not found
        not_found_target = target_paths[..., 0] == 0
        if torch.any(not_found_target):
            is_not_loop = (row_indices[:, None] != col_matrix.reshape(col_matrix.shape[0], -1)).reshape(
                    col_matrix.shape)
            not_found_target = not_found_target & is_not_loop

        target_distances = torch.where(not_found_target, self._get_default_distance(), target_distances)

        if parameters['k_nearest'] != 0:
            nearest_paths = torch.as_tensor(np.copy(nearest_paths), dtype=torch.int64)
            nearest_distances = F.softplus(self._get_logits(self.edge_weight_logits, nearest_paths)).sum(dim=(-1))
            nearest_vertices = edge_indices[nearest_paths[..., 0]]
            # ^--[batch_size, k_nearest]
        else:
            nearest_paths = nearest_distances = nearest_vertices = None

        return dict(
                target_paths=target_paths,
                target_distances=target_distances,
                logp_target_paths=logp_target_paths,
                found_target=~not_found_target,

                nearest_paths=nearest_paths,
                nearest_distances=nearest_distances,
                nearest_vertices=nearest_vertices,
        )
