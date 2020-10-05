"""
Baseline GloVe as in https://github.com/maciejkula/glove-python/
"""
from warnings import warn

import numba
import glove

import numpy as np
import torch
import glove


class GloveDataset:
    def __init__(self, corpora: glove.Corpus, *, lowercase, alpha=0.75, max_count=100, max_targets=10_000):
        """
        Helper class to train glove model on a pre-computed cooccurence matrix
        :type corpora: glove.Corpus
        :param alpha: glove weighting function slope
        :param max_count: co-occurence threshold, after which all words have weight=1
        :param max_targets: samples (up to) this many targets in a batch
        """
        corpora_contains_uppercase = not all(map(lambda w: w.islower() or not w.isalpha(), corpora.dictionary.keys()))

        if lowercase is True:
            assert not corpora_contains_uppercase, "lowercase=True but corpora.dictionary contains uppercase tokens"
        elif lowercase is False:
            if not corpora_contains_uppercase:
                warn("lowercase=False but there are no uppercase words in corpora.dictionary. Did you forget to set it?")
        elif lowercase is None:
            lowercase = not corpora_contains_uppercase
            warn("Automatically inferred lowercase={}".format(lowercase))

        super().__init__()
        self.alpha, self.max_count, self.max_targets = alpha, max_count, max_targets
        self.lowercase = lowercase

        self.corpora = corpora
        self.cooc_matrix = (corpora.matrix + corpora.matrix.T).tocsr()
        self.cooc_matrix.sort_indices()

        weight_matrix = self.cooc_matrix.copy()
        weight_matrix.data = self.cooc_to_weight(weight_matrix.data)
        weight_matrix.sort_indices()

        self.row_total_weights = np.asarray(weight_matrix.sum(axis=-1)).reshape(-1)
        self.row_num_nonzeroes = np.diff(weight_matrix.indptr)
        self.row_probs = self.row_total_weights / np.sum(self.row_total_weights)

        self.token_to_ix = dict(corpora.dictionary)
        self.ix_to_token = {i: token for token, i in self.token_to_ix.items()}

    def cooc_to_weight(self, cooc):
        return np.minimum(1.0, (cooc / self.max_count)) ** self.alpha

    @staticmethod
    @numba.jit(nopython=True)
    def _sample_nonzeroes(matrix_values, matrix_indptr, matrix_indices, max_elems_per_row=10_000):
        """
        Sample up to :max_elems_per_row: nonzero indices for each row 
        in sparse csr matrix defined by indptr and indices
        """
        num_rows = len(matrix_indptr) - 1
        output_indices = np.full((num_rows, max_elems_per_row), -1, dtype=np.int32)
        output_values = np.full((num_rows, max_elems_per_row), 0, dtype=np.float32)

        for row_i in range(num_rows):
            indices = matrix_indices[matrix_indptr[row_i]: matrix_indptr[row_i + 1]]
            values = matrix_values[matrix_indptr[row_i]: matrix_indptr[row_i + 1]]

            if len(indices) > max_elems_per_row:
                selector = np.random.choice(len(indices), replace=False, size=max_elems_per_row)
                indices = indices[selector]
                values = values[selector]

            output_indices[row_i, :len(indices)] = indices
            output_values[row_i, :len(values)] = values
        return output_indices, output_values

    def form_batch(self, batch_ii=None, batch_size=None, replace=False, max_elems_per_row=None):
        """ Sample training batch for given rows """
        assert (batch_ii is None) != (batch_size is None), "please provide either batch_ii or batch_size but not both"
        max_elems_per_row = max_elems_per_row or self.max_targets
        if batch_ii is None:
            batch_ii = np.random.choice(len(self.row_probs), size=batch_size, replace=replace, p=self.row_probs)

        batch_ii_repeated = np.repeat(batch_ii[:, None], repeats=max_elems_per_row, axis=1)
        cooc_rows = self.cooc_matrix[batch_ii]
        batch_jj, batch_cooc = self._sample_nonzeroes(cooc_rows.data, cooc_rows.indptr, cooc_rows.indices,
                                                      max_elems_per_row=max_elems_per_row or self.max_targets)
        batch_jj_mask = batch_jj != -1
        batch_jj_masked = np.where(batch_jj_mask, batch_jj, batch_ii_repeated)

        glove_weights = self.cooc_to_weight(batch_cooc)
        sample_weights = glove_weights / (glove_weights * batch_jj_mask).sum(-1, keepdims=True)

        sample_weights = np.where(batch_jj_mask, sample_weights, 0)
        targets = np.log(np.where(batch_jj_mask, batch_cooc, 1))

        batch = dict(
            row_indices=batch_ii, col_matrix=batch_jj_masked, mask=batch_jj_mask, cooc=batch_cooc,
            glove_weights=glove_weights, sample_weights=sample_weights, targets=targets
        )
        return {key: torch.as_tensor(value) for key, value in batch.items()}

    def iterate_minibatches(self, *args, total_batches=float('inf'), **kwargs):
        batches_so_far = 0
        while batches_so_far < total_batches:
            yield self.form_batch(*args, **kwargs)
            batches_so_far += 1
