import os
import argparse
import sys

import torch

sys.path.insert(0, os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2]))
from lib.task.nlp import evaluate_analogy



def evaluate_analogies(args):
    model = torch.load(args.path)
    metrics = evaluate_analogy(model, lowercase=args.lowercase)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    parser.add_argument('--not_lowercase', action='store_false', dest='lowercase')
    parsed_args = parser.parse_args()
    evaluate_analogies(parsed_args)
