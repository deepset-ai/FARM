import random
import numpy as np
import torch
import logging


logger = logging.getLogger(__name__)


def set_all_seeds(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def initialize_device_settings(no_cuda, local_rank, fp16):
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        if(device == torch.device("cpu")):
            n_gpu = 0
        else:
            n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))
    return device, n_gpu