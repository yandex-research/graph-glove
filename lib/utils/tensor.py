import contextlib
import gc
import os
import time
from collections import Counter
from itertools import chain
from warnings import warn
import numpy as np

import torch
from torch import nn as nn


def nop(x):
    return x


@contextlib.contextmanager
def nop_ctx():
    yield None


class Nop(nn.Module):
    def forward(self, x):
        return x


class Residual(nn.Sequential):
    def forward(self, x):
        return super().forward(x) + x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)


@contextlib.contextmanager
def training_mode(*modules, is_train: bool):
    group = nn.ModuleList(modules)
    was_training = {module: module.training for module in group.modules()}
    try:
        yield group.train(is_train)
    finally:
        for key, module in group.named_modules():
            if module in was_training:
                module.training = was_training[module]
            else:
                raise ValueError("Model was modified inside training_mode(...) context, could not find {}".format(key))


def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)


def infer_model_device(model: nn.Module):
    """ infers model device as the device where the majority of parameters and buffers are stored """
    device_stats = Counter(
        tensor.device for tensor in chain(model.parameters(), model.buffers())
        if torch.is_tensor(tensor)
    )
    return max(device_stats, key=device_stats.get)


class Lambda(nn.Module):
    def __init__(self, func):
        """ :param func: call this function during forward """
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


if run_from_ipython():
    from IPython.display import clear_output
else:
    def clear_output(*args, **kwargs):
        os.system('clear')


try:
    import numba
    maybe_jit = numba.jit
except Exception as e:
    warn("numba not found or failed to import, some operations may run slowly;\n"
         "Error message: {}".format(e))
    maybe_jit = nop


@maybe_jit
def sliced_argmax(inp, slices, out=None):
    if out is None:
        out = np.full(len(slices) - 1, -1, dtype=np.int64)
    for i in range(len(slices) - 1):
        if slices[i] == slices[i + 1]: continue
        out[i] = np.argmax(inp[slices[i]: slices[i + 1]])
    return out

