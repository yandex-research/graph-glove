"""
Worker that finds paths for the training procedure
"""
import argparse
import os
import sys
import time
from traceback import print_exc
# setup path to allow importing lib
sys.path.insert(0, os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2]))

from lib.distributed_glove.worker import parallel_worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--n_threads', type=int, required=True)
    parser.add_argument('--model_update_interval', type=int, default=1)
    parser.add_argument('--restart_on_error', dest='restart_on_error', default=False, action='store_true')
    args = parser.parse_args()

    while True:
        try:
            parallel_worker(args.host, args.port, args.password, args.model_update_interval, args.n_threads)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            exit()
        except BaseException as e:
            print_exc()
            time.sleep(1)
            if not args.restart_on_error:
                exit()

