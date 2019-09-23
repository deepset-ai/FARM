import logging
import random

import numpy as np
import torch
import mlflow
from copy import deepcopy
from farm.visual.ascii.images import WELCOME_BARN, WORKER_M, WORKER_F, WORKER_X

logger = logging.getLogger(__name__)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    logger.warning("TensorboardX not installed. Required if you use tensorboard logger.")


def set_all_seeds(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def initialize_device_settings(use_cuda, local_rank=-1, fp16=False):
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

    def __init__(self, tracking_uri, **kwargs):
        self.tracking_uri = tracking_uri
        print(WELCOME_BARN)

    def init_experiment(self, tracking_uri):
        raise NotImplementedError()

    @classmethod
    def log_metrics(cls, metrics, step):
        raise NotImplementedError()

    @classmethod
    def log_artifacts(cls, self):
        raise NotImplementedError()

    @classmethod
    def log_params(cls, params):
        raise NotImplementedError()


class MLFlowLogger(BaseMLLogger):
    """
    Logger for MLFlow experiment tracking.
    """

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name, nested=nested)

    @classmethod
    def log_metrics(cls, metrics, step):
        mlflow.log_metrics(metrics, step=step)

    @classmethod
    def log_params(cls, params):
        mlflow.log_params(params)

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        mlflow.log_artifacts(dir_path, artifact_path)


class TensorBoardLogger(BaseMLLogger):
    """
    PyTorch TensorBoard Logger
    """

    def __init__(self, **kwargs):
        TensorBoardLogger.summary_writer = SummaryWriter()
        super().__init__(**kwargs)

    @classmethod
    def log_metrics(cls, metrics, step):
        for key, value in metrics.items():
            TensorBoardLogger.summary_writer.add_scalar(
                tag=key, scalar_value=value, global_step=step
            )

    @classmethod
    def log_params(cls, params):
        for key, value in params.items():
            TensorBoardLogger.summary_writer.add_text(tag=key, text_string=str(value))


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def convert_iob_to_simple_tags(preds, spans):
    simple_tags = []
    merged_spans = []
    open_tag = False
    for pred, span in zip(preds, spans):
        # no entity
        if not ("B-" in pred or "I-" in pred):
            if open_tag:
                # end of one tag
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                open_tag = False
            continue

        # new span starting
        elif "B-" in pred:
            if open_tag:
                # end of one tag
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
            cur_tag = pred.replace("B-", "")
            cur_span = span
            open_tag = True

        elif "I-" in pred:
            this_tag = pred.replace("I-", "")
            if open_tag and this_tag == cur_tag:
                cur_span["end"] = span["end"]
            elif open_tag:
                # end of one tag
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                open_tag = False
    if open_tag:
        merged_spans.append(cur_span)
        simple_tags.append(cur_tag)
        open_tag = False
    return simple_tags, merged_spans


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist

def log_ascii_workers(n, logger):
    m_worker_lines = WORKER_M.split("\n")
    f_worker_lines = WORKER_F.split("\n")
    x_worker_lines = WORKER_X.split("\n")
    all_worker_lines = []
    for i in range(n):
        rand = np.random.randint(low=0,high=3)
        if(rand % 3 == 0):
            all_worker_lines.append(f_worker_lines)
        elif(rand % 3 == 1):
            all_worker_lines.append(m_worker_lines)
        else:
            all_worker_lines.append(x_worker_lines)
    zipped = zip(*all_worker_lines)
    for z in zipped:
        logger.info("  ".join(z))

def format_log(ascii, logger):
    ascii_lines = ascii.split("\n")
    for l in ascii_lines:
        logger.info(l)

