from time import time, sleep

import torch
from prefetch_generator import BackgroundGenerator
from torch.optim import SparseAdam, Adam

from lib.distributed_glove import GloveModel
from lib.graph_embedding import inverse_sigmoid
from .glove_dataset import DistributedGloveDataset
from lib.utils import check_numpy, JobServer, training_mode, infer_model_device
from lib.utils.base_trainer import BaseTrainer
from lib.task.nlp import evaluate_similarity


class DistributedTrainer(BaseTrainer):
    def __init__(self, model: GloveModel, dataset: DistributedGloveDataset, job_server: JobServer,
                 report_frequency=50, upload_frequency=1, OptimizerClass=None, similarity_benchmarks=('SCWS', 'WS353'),
                 base_lambda=10.0, lambda_warmup=1000, reg_batch_size=16384, clip_prob_max=None,
                 optimizer_kwargs=None, **kwargs):
        """ A special class that handles GloVe training on minibatches """

        if OptimizerClass is None:
            OptimizerClass = SparseAdam if model.graph_embedding.sparse else Adam
        optimizer = OptimizerClass(model.parameters(), **optimizer_kwargs or {})

        super().__init__(model=model, optimizer=optimizer, **kwargs)
        self.dataset = dataset
        self.job_server = job_server
        self.upload_graph(is_train=True)
        self.report_frequency = report_frequency
        self.upload_frequency = upload_frequency
        self.prev_batch_time = time()
        self.base_lambda, self.lambda_warmup, self.reg_batch_size = base_lambda, lambda_warmup, reg_batch_size
        self.clip_prob_max = clip_prob_max
        self.similarity_benchmarks = similarity_benchmarks

    def train_on_batch(self, batch):
        if self.verbose:
            print(f"Received new batch in {time() - self.prev_batch_time:.3}s")
        self.prev_batch_time = time()
        device = infer_model_device(self)

        with training_mode(self, is_train=True):
            graph_emb = self.model.graph_embedding
            total_edges = graph_emb.num_edges

            prediction, mean_logp_path = self.model(**batch)

            squared_error = (batch['targets'].to(device) - prediction) ** 2
            loss = (squared_error * batch['sample_weights'].to(device)).sum() / batch['sample_weights'].to(device).sum()
            # Note to future self: we had no idea if one _should_ divide by batch['sample_weights'] in this case

            regularizer = graph_emb.compute_l0_prior_penalty(
                batch_size=self.reg_batch_size, free_edge=True) * (graph_emb.num_edges / total_edges)

            lambd = self.base_lambda * min(1.0, self.total_steps / self.lambda_warmup)
            obj = loss - mean_logp_path + lambd * regularizer

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()
            if self.clip_prob_max is not None:
                with torch.no_grad():
                    max_logit = inverse_sigmoid(self.clip_prob_max)
                    graph_emb.edge_weight_logits.clamp_max_(max_logit)

        print(f'Step #{self.total_steps}, Obj: {obj.item():.3}, '
              f'Time to process batch: {time() - self.prev_batch_time:.3}s')

        if self.total_steps % self.upload_frequency == 0:
            self.upload_graph(is_train=True)

        self.prev_batch_time = time()

        return self.record(prefix='train/', step=self.total_steps,
                           obj=obj.item(), loss=loss.item(), reg=regularizer.item(),
                           mean_logp_path=mean_logp_path.item())

    def evaluate_metrics(self, *args, **kwargs):
        metrics = evaluate_similarity(self.model, lowercase=self.dataset.lowercase, datasets=self.similarity_benchmarks)
        return self.record(prefix='val/similarity/', step=self.total_steps,
                           params_per_vertex=self.model.graph_embedding.report_model_size()['params_per_vertex'],
                           **metrics)

    def upload_graph(self, is_train):
        """ Uploads graph embedding unto JobServer for workers to access """
        if self.verbose:
            print(end="Uploading graph to job server... ")
        with training_mode(self, is_train=is_train):
            device = infer_model_device(self.model.graph_embedding)
            self.job_server.save_model(self.model.graph_embedding.cpu())
            self.model.graph_embedding.to(device)
        if self.verbose:
            print("Done!")

    def fit(self, *, batch_size=64, buffer_size_multiple=2, prefetch_multiple=10, epochs=float('inf'), **kwargs):
        """ Legacy: """
        buffer_size = buffer_size_multiple * batch_size
        batch_iterator = self.dataset.iterate_minibatches(
            model=self.model, batch_size=batch_size, buffer_size=buffer_size,
            null_vertex=self.model.null_vertex, num_vertices=self.model.graph_embedding.num_vertices)
        return super().fit(
            training_data=BackgroundGenerator(batch_iterator, max_prefetch=prefetch_multiple * buffer_size_multiple),
            batches_per_epoch=self.report_frequency, epochs=epochs,
            validate=True, **kwargs,
        )

    def save_checkpoint(self, *args, **kwargs):
        """ Save as usual, also dump graph structure """
        path = super().save_checkpoint(*args, **kwargs)
        if self.verbose:
            path_to_model = path
            if path_to_model.endswith('.pth'):
                path_to_model = path_to_model[:-4]
            path_to_model = path_to_model + ".model.pt"
            print(f"Saving model to {path_to_model}")
            torch.save(self.model, path_to_model)
        return path
