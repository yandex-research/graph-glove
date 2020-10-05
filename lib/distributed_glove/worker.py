import logging
import time
from multiprocessing import cpu_count
from threading import Thread
from traceback import print_exc

from lib.utils.threading import AtomicCounter, SharedStorage
from lib.utils.job_server import JobServer, JobServerOutOfDateException


def parallel_worker(host, port, password, model_update_interval=10, n_threads=-1, **kwargs):
    """ Main worker method. Acquires tasks, runs them in parallel and submits results at ready """
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    if n_threads < 0:
        n_threads = cpu_count() - n_threads + 1

    db = JobServer(host=host, port=port, password=password, **kwargs)

    update_cnt = AtomicCounter()
    model_storage = SharedStorage()

    model = db.load_model()
    model_storage.set(dict(model=model, dijkstra_parameters=model.prepare_for_dijkstra()))

    threads = [Thread(target=worker_thread_function,
                      args=(db, model_storage, update_cnt, thread_number))
               for thread_number in range(n_threads)]

    try:
        for thread in threads:
            thread.start()

        while True:
            update_cnt.wait_until_reaches(model_update_interval * n_threads)
            update_cnt.reset()

            logging.info('Updating model')
            start = time.time()
            model = db.load_model()
            model_storage.set(dict(model=model, dijkstra_parameters=model.prepare_for_dijkstra()))
            logging.info(f'Model updated in {time.time() - start:.3}s')
    except JobServerOutOfDateException as e:
        print_exc()
        logging.info("Terminated parallel_worker due to end of epoch (indicated by JobServer version change).")
    finally:
        logging.info("Waiting for threads to terminate...")
        for thread in threads:
            thread.join()
        logging.info("Done!")


def worker_thread_function(db: JobServer, storage: SharedStorage, update_cnt: AtomicCounter, thread_num: int):
    logging.info("Starting worker {}".format(thread_num))
    snapshot = storage.get()
    while True:
        start_time = time.time()
        try:
            task = db.get_job()
            end_time = time.time()
            snapshot = storage.update(snapshot)
            model = snapshot.data['model']
            dijkstra_parameters = snapshot.data['dijkstra_parameters']
            logging.info(f"Worker {thread_num}: got job in {end_time - start_time:.3} seconds")
            start_time = time.time()
            row_indices = task.pop('row_indices')
            col_matrix = task.pop('col_matrix')
            result = model.compute_paths(row_indices, col_matrix, dijkstra_parameters, **task,
                                         presample_edges=False, n_jobs=1)
            db.commit_result(result)
            update_cnt.increment()
            end_time = time.time()
            logging.info(f"Worker {thread_num}: processed job in {end_time - start_time:.3} seconds")

        except JobServerOutOfDateException as e:
            logging.info(f"Worker {thread_num} terminated due to version update:")
            print_exc()
            update_cnt.increment(float('inf'))  # cause worker to terminate batch immediately
            return

        except Exception as e:
            logging.info(f"Worker {thread_num} caught exception:")
            print_exc()
            update_cnt.increment(float('inf'))  # cause worker to terminate batch immediately
            logging.info(f"Worker {thread_num} will now restart.")
