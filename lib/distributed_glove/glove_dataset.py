import sys

import torch
from uuid import uuid4
import numpy as np
from glove import Corpus

from ..task.nlp import GloveDataset
from ..utils import infer_model_device, defaultdict, check_numpy, JobServer
from . import GloveModel

JOB_TYPE_REGULAR, JOB_TYPE_NORMS = "JOB_TYPE_REGULAR", "JOB_TYPE_NORMS"


class DistributedGloveDataset(GloveDataset):
    def __init__(self, *, corpora: Corpus, job_server: JobServer, verbose=True, **kwargs):
        super().__init__(corpora, **kwargs)
        self.job_server = job_server
        self.verbose = verbose

    def iterate_minibatches(self, *, batch_size, buffer_size, job_chunk_size=None, total_batches=float('inf'),
                            model: GloveModel, **kwargs):
        """
        Generate minibatches of data points using worker pool
        Each batch contains a few unique rows and potentially many columns per row (see GloveDataset)
        :param batch_size: number of COOC rows in batch
        :param buffer_size: number of unfinished batches that can exist simultaneously.
            Smaller buffer = less that paths found by workers are no longer optimal for trainer (due to sgd updates)
        :param job_chunk_size: if specified, dispatches worker jobs in chunks of this size (default to batch_size)
        :param total_batches:  if specified, terminates after this many batches
        :type model: GloveModel
        """
        ids = dict()  # registry of created jobs. A job is put into registry at birth and removed at completion

        null_vertex = model.null_vertex
        num_vertices = model.graph_embedding.num_vertices
        device = infer_model_device(model)

        def create_norms_job():

            sample_id = uuid4().int
            job_inputs = {
                'row_indices': np.array([null_vertex], dtype='int32'),
                'col_matrix': np.arange(num_vertices, dtype='int32')[None],
                'sample_id': sample_id, **kwargs,
            }
            ids[job_inputs['sample_id']] = {"sample_id": sample_id, "type": JOB_TYPE_NORMS}
            return job_inputs

        # (1) create a job for initial norms and wait for it to complete
        paths_from_v0 = None
        self.job_server.add_jobs(create_norms_job())
        while paths_from_v0 is None:
            job_result = self.job_server.get_result()
            if ids[job_result['sample_id']].get("type") == JOB_TYPE_NORMS:
                paths_from_v0 = job_result
            elif self.verbose:
                print(f'Found unknown sample_id {job_result.get("sample_id")}', file=sys.stderr)

        # (2) generate regular jobs, with occasional norm jobs inbetween
        def job_generator():
            total_samples = 0
            while True:
                batch = self.form_batch(batch_size=batch_size)
                keys_to_send = ('row_indices', 'col_matrix')
                keys_to_keep = [key for key in batch.keys() if key not in keys_to_send]
                for elem in range(batch_size):
                    # submit normal jobs
                    sample_id = uuid4().int
                    data_to_send = {key: check_numpy(batch[key][[elem]]) for key in keys_to_send}
                    data_to_keep = {key: batch[key][[elem]] for key in keys_to_keep}
                    ids[sample_id] = {'sample_id': sample_id, "type": JOB_TYPE_REGULAR, **data_to_keep}
                    yield {'sample_id': sample_id, **data_to_send, **kwargs}

                    total_samples += 1
                    if total_samples >= total_batches:
                        break

                # submit job to compute norms
                yield create_norms_job()

        # (3) assemble regular job results into batches, update norms as they appear
        def postprocess_batch(batch):
            agg_results = defaultdict(list)
            for result in batch:
                sample_id = result['sample_id']
                for key, value in result.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        agg_results[key].append(torch.from_numpy(check_numpy(value)))
                for key, value in ids[sample_id].items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        agg_results[key].append(value)
                del ids[sample_id]

            concat_result = {key: torch.cat(value).to(device) for key, value in agg_results.items()}
            concat_result['vertex_norms_output'] = paths_from_v0
            return concat_result

        current_batch = []
        for results_chunk in self.job_server.iterate_minibatches(
            job_generator(), job_chunk_size or batch_size, buffer_size=buffer_size):

            for job_result in results_chunk:
                sample_id = job_result['sample_id']
                if sample_id not in ids:
                    print(f'Found unknown sample_id {sample_id}', file=sys.stderr)
                    continue
                elif ids[sample_id]['type'] == JOB_TYPE_NORMS:
                    paths_from_v0 = job_result
                    del ids[sample_id]

                elif ids[sample_id]['type'] == JOB_TYPE_REGULAR:
                    current_batch.append(job_result)

                    if len(current_batch) >= batch_size:
                        yield [postprocess_batch(current_batch)]  # note: we use list to make sure batch isn't *unpacked
                        current_batch = []

                else:
                    print(f'Found unknown job type {ids[sample_id]["type"]}', file=sys.stderr)
