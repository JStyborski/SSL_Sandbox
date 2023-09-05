import math
import datetime
import numpy as np

import psutil

# Not used here, but in the calling script
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from fvcore.nn.parameter_count import parameter_count

import torch

# Params counter from old JStyborski code
def count_params(model):
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    allParams = sum(p.numel() for p in model.parameters())
    return trainableParams, allParams
#print(count_params(model))

# Code from https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/misc.py

def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))

def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, torch.nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total

# The following code is for using fvcore modules

def sample_input(batchLoader):
    firstBatch, _ = next(iter(batchLoader))
    del batchLoader
    return firstBatch

#activCount = activation_count(model, inputsTuple) # Records # of activations (in millions) for each layer type
#flopCount = flop_count(model, inputsTuple) # Records # of gflops for each layer type
#paramCount = parameter_count(model, inputsTuple) # Records # of parameters per layer

# All above have _table methods to output a pretty table - e.g., flop_count_table()

# Here's working sample code from running in SimSiam

#trainableParamsJS, allParamsJS = count_params(model)
#paramCountSF = params_count(model)
#gpuUsage = gpu_mem_usage()
#cpuUsage, cpuTotal = cpu_mem_usage()
#sampleInput = sample_input(train_loader)
#if args.gpu is not None:
#    sampleInput[0] = sampleInput[0].cuda(args.gpu, non_blocking=True)
#    sampleInput[1] = sampleInput[1].cuda(args.gpu, non_blocking=True)
#zap = activation_count(model, (sampleInput[0], sampleInput[1]))
#zep = flop_count(model, (sampleInput[0], sampleInput[1]))
#zip = parameter_count(model)
