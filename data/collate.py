import numpy as np
import random
import torch

def collate_fn_test(batch):
    batch = zip(*batch)
    outputs = []
    for output in batch:
        outputs.append(np.concatenate(output, 0))
    return tuple(outputs)

def collate_fn_train(batch):
    batch = zip(*batch)
    outputs = []
    for output in batch:
        outputs.append(np.concatenate(output, 0))
    return tuple(outputs)

def worker_init_fn(worker_id, seed=0):
    worker_seed = worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
