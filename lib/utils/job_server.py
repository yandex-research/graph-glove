from threading import Event

import redis
import os, sys, time
from warnings import warn
import pickle
from multiprocessing import BoundedSemaphore
from prefetch_generator import background
from .tensor import nop


class JobServer:
    def __init__(self,
                 host="localhost",
                 port=7070,
                 jobs_key="jobs",
                 results_key="results",
                 model_key="model",
                 worker_prefix="workers.",
                 version_key="version",
                 version_lock="check_version",
                 default_version=0,
                 redis_init_wait_seconds=5,
                 check_version_frequency=1,
                 password=None,
                 verbose=True,
                 **kwargs,
                 ):
        """
        The database instance that can stores paths, model weights and metadata.
        Implemented as a thin wrapper over Redis.

        :param host: redis hostname
        :param port: redis port
        :param args: args for database client (redis)
        :param kwargs: kwargs for database client (redis), e.g. password="***"
        :param default_prefix: prepended to both default_params_key and default_session_key and
            default_worker_prefix. Does NOT affect custom keys.
        :param jobs_key: default name for incomplete job
        :param results_key: default name for job results
        :param model_key: default name for weights pickle
        """
        self.model_key, self.jobs_key, self.results_key = model_key, jobs_key, results_key
        self.worker_prefix, self.version_key, self.version_lock = worker_prefix, version_key, version_lock
        self.verbose = verbose

        # if localhost and can't find redis, start one
        if host in ("localhost", "0.0.0.0", "127.0.0.1", "*"):
            try:
                redis.Redis(host=host, port=port, password=password, **kwargs).client_list()
            except redis.ConnectionError:
                # if not, on localhost try launch new one
                if self.verbose:
                    print("Redis not found on %s:%s. Launching new redis..." % (host, port))
                self.start_redis(port=port, requirepass=password, **kwargs)
                time.sleep(redis_init_wait_seconds)

        self.redis = redis.Redis(host=host, port=port, password=password, **kwargs)
        if self.verbose and len(self.redis.keys()):
            warn("Found existing keys: {}".format(self.redis.keys()))

        if self.version_key not in self.redis.keys():
            with self.redis.lock(self.version_lock):
                self.redis.set(self.version_key, default_version)

        self.local_version = int(self.redis.get(self.version_key))
        self.check_version_frequency = check_version_frequency

    def start_redis(self, **kwargs):
        """starts a redis serven in a NON-DAEMON mode"""
        kwargs_list = [
            "--{} {}".format(name, value)
            for name, value in kwargs.items()
        ]
        cmd = "nohup redis-server {} > .redis.log &".format(' '.join(kwargs_list))
        if self.verbose:
            print("CMD:", cmd)
        os.system(cmd)

    def check_version(self):
        """ Verify that local_version matches remote version, otherwise raise JobServerOutOfDateException """
        remote_version = int(self.redis.get(self.version_key))
        if self.local_version < remote_version:
            raise JobServerOutOfDateException("JobServer version is {} while local version is {}."
                                              "Please call self.update_version() or create new JobServer client"
                                              "".format(remote_version, self.local_version))
        return remote_version

    def update_version(self, *, by=None, new_version=None, synchronize=True, await_seconds=0):
        """
        Manipulate versions: call self.update_version() to load remote version, self.update_version(by=i) to increment,
        self.update_version(new_version=1337) to set new version exactly, synchronize=False to only affect local_version
        """
        assert by is None or new_version is None, "please provide either by or new_version or neither, not both"
        current_version = int(self.redis.get(self.version_key)) if synchronize else self.local_version
        if new_version is None:
            new_version = current_version + int(by or 0)
        self.local_version = new_version
        if synchronize:
            with self.redis.lock(self.version_lock):
                self.redis.set(self.version_key, new_version)
        if await_seconds:
            time.sleep(await_seconds)
        return new_version

    def reset_queue(self):
        for key in self.jobs_key, self.results_key:
            self.redis.delete(key)
        return True

    @staticmethod
    def dumps(data):
        """ converts whatever to string """
        return pickle.dumps(data, protocol=4)

    @staticmethod
    def loads(string):
        """ converts string to whatever was dumps'ed in it """
        return pickle.loads(string)

    def save_model(self, model):
        self.check_version()
        self.redis.set(self.model_key, self.dumps(model))
        return model

    def load_model(self):
        self.check_version()
        payload = self.redis.get(self.model_key)
        if payload is None:
            raise ValueError("Failed to load model: no such key ({})".format(self.model_key))
        return self.loads(payload)

    def add_jobs(self, *jobs):
        self.check_version()
        return self.redis.rpush(self.jobs_key, *map(self.dumps, jobs))

    def get_job(self):
        while True:
            self.check_version()
            payload = self.redis.blpop(self.jobs_key, timeout=self.check_version_frequency)
            if payload is None: continue
            return self.loads(payload[1])

    def commit_result(self, result):
        self.check_version()
        return self.redis.rpush(self.results_key, self.dumps(result))

    def get_result(self):
        while True:
            self.check_version()
            payload = self.redis.blpop(self.results_key, timeout=self.check_version_frequency)
            if payload is None: continue
            return self.loads(payload[1])

    def get_results_batch(self, batch_size):
        # get up to batch_size first elements
        results = list(map(self.loads, self.redis.lrange(self.results_key, 0, batch_size - 1)))
        self.redis.ltrim(self.results_key, len(results), -1)
        for i in range(len(results), batch_size):
            results.append(self.get_result())
        return results

    def iterate_minibatches(self, generate_jobs, batch_size, buffer_size, postprocess_batch=None):
        assert buffer_size >= batch_size
        buffer_semaphore = BoundedSemaphore(buffer_size)  # jobs currently in buffer
        generate_jobs = iter(generate_jobs)
        postprocess_batch = postprocess_batch or nop

        @background(max_prefetch=-1)
        def _load_jobs_and_iterate_batch_sizes():
            """
            Loads jobs into queue, yields batch_size every time a generator
            can order this batch_size from the database
            """
            current_batch = 0
            for task in generate_jobs:
                buffer_semaphore.acquire()
                self.add_jobs(task)
                current_batch += 1
                if current_batch == batch_size:
                    yield current_batch  # you can now load additional batch_size elements
                    current_batch = 0
            if current_batch != 0:
                yield current_batch  # load the remaining elements

        for allowed_batch_size in _load_jobs_and_iterate_batch_sizes():
            batch = self.get_results_batch(allowed_batch_size)
            for _ in batch:
                buffer_semaphore.release()

            yield postprocess_batch(batch)


class JobServerOutOfDateException(BaseException):
    pass
