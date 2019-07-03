import logging
import random

import numpy as np
import torch
from mlflow import (
    log_metrics,
    log_params,
    set_tracking_uri,
    set_experiment,
    start_run,
    log_artifacts,
)

logger = logging.getLogger(__name__)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    logger.warn("TensorboardX not installed. If you use tensordoard logger.")


def set_all_seeds(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def initialize_device_settings(use_cuda, local_rank, fp16):
    if not use_cuda:
        device = torch.device("cpu")
        n_gpu = 0
    elif local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            n_gpu = 0
        else:
            n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(local_rank != -1), fp16
        )
    )
    return device, n_gpu


class BaseMLLogger:
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLFlow, Tensorboard, or Sacred.
    """

    def __init__(self, experiment_name, uri, **kwargs):
        self.experiment_name = experiment_name
        self.uri = uri

    def init_experiment(self, tracking_uri):
        raise NotImplementedError()

    @classmethod
    def init_trail(cls, trail_name, nested=True):
        raise NotImplementedError()

    @classmethod
    def write_metrics(cls, metrics, step):
        raise NotImplementedError()

    @classmethod
    def add_artifacts(cls, self):
        raise NotImplementedError()

    @classmethod
    def write_params(cls, params):
        raise NotImplementedError()


class MLFlowLogger(BaseMLLogger):
    """
    Logger for MLFlow experiment tracking.
    """

    def init_experiment(self, tracking_uri):
        set_experiment(self.experiment_name)
        set_tracking_uri(tracking_uri)

    @classmethod
    def init_trail(cls, trail_name, nested=True):
        start_run(run_name=trail_name, nested=nested)

    @classmethod
    def write_metrics(cls, metrics, step):
        log_metrics(metrics, step=step)

    @classmethod
    def write_params(cls, params):
        log_params(params)

    @classmethod
    def add_artifacts(cls, dir_path, artifact_path=None):
        log_artifacts(dir_path, artifact_path)


class TensorBoardLogger(BaseMLLogger):
    """
    PyTorch TensorBoard Logger
    """

    def __init__(self, **kwargs):
        TensorBoardLogger.summary_writer = SummaryWriter()
        super().__init__(**kwargs)

    @classmethod
    def write_metrics(cls, metrics, step):
        for key, value in metrics.items():
            TensorBoardLogger.summary_writer.add_scalar(
                tag=key, scalar_value=value, global_step=step
            )

    @classmethod
    def write_params(cls, params):
        for key, value in params.items():
            TensorBoardLogger.summary_writer.add_text(tag=key, text_string=str(value))
