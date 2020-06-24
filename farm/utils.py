import hashlib
import json
import logging
import random
import os
import signal
import numpy as np
import torch
from requests.exceptions import ConnectionError
from torch import multiprocessing as mp
import mlflow
from copy import deepcopy
import pandas as pd
from tqdm import tqdm


from farm.visual.ascii.images import WELCOME_BARN, WORKER_M, WORKER_F, WORKER_X


logger = logging.getLogger(__name__)


def set_all_seeds(seed, deterministic_cudnn=False):
    """
    Setting multiple seeds to make runs reproducible.

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn) !

    :param seed:number to use as seed
    :type seed: int
    :param deterministic_torch: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    :type deterministic_cudnn: bool
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_chunksize(num_dicts, min_chunksize=4, max_chunksize=2000, max_processes=128):
    num_cpus = min(mp.cpu_count() - 1 or 1, max_processes)  # -1 to keep a CPU core free for the main process
    dicts_per_cpu = np.ceil(num_dicts / num_cpus)
    # automatic adjustment of multiprocessing chunksize
    # for small files (containing few dicts) we want small chunksize to ulitize all available cores but never less
    # than 2, because we need it to sample another random sentence in LM finetuning
    # for large files we want to minimize processor spawning without giving too much data to one process, so we
    # clip it at 5k
    multiprocessing_chunk_size = int(np.clip((np.ceil(dicts_per_cpu / 5)), a_min=min_chunksize, a_max=max_chunksize))
    # This lets us avoid cases in lm_finetuning where a chunk only has a single doc and hence cannot pick
    # a valid next sentence substitute from another document
    if num_dicts != 1:
        while num_dicts % multiprocessing_chunk_size == 1:
            multiprocessing_chunk_size -= -1
    dict_batches_to_process = int(num_dicts / multiprocessing_chunk_size)
    num_processes = min(num_cpus, dict_batches_to_process) or 1

    return multiprocessing_chunk_size, num_processes


def initialize_device_settings(use_cuda, local_rank=-1, use_amp=None):
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
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, automatic mixed precision training: {}".format(
            device, n_gpu, bool(local_rank != -1), use_amp
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


class StdoutLogger(BaseMLLogger):
    """ Minimal logger printing metrics and params to stdout.
    Useful for services like AWS SageMaker, where you parse metrics from the actual logs"""

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        logger.info(f"\n **** Starting experiment '{experiment_name}' (Run: {run_name})  ****")

    @classmethod
    def log_metrics(cls, metrics, step):
        logger.info(f"Logged metrics at step {step}: \n {metrics}")

    @classmethod
    def log_params(cls, params):
        logger.info(f"Logged parameters: \n {params}")

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        raise NotImplementedError

    @classmethod
    def end_run(cls):
        logger.info(f"**** End of Experiment **** ")


class MLFlowLogger(BaseMLLogger):
    """
    Logger for MLFlow experiment tracking.
    """

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name, nested=nested)
        except ConnectionError:
            raise Exception(
                f"MLFlow cannot connect to the remote server at {self.tracking_uri}.\n"
                f"MLFlow also supports logging runs locally to files. Set the MLFlowLogger "
                f"tracking_uri to an empty string to use that."
            )

    @classmethod
    def log_metrics(cls, metrics, step):
        try:
            mlflow.log_metrics(metrics, step=step)
        except ConnectionError:
            logger.warning(f"ConnectionError in logging metrics to MLFlow.")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    @classmethod
    def log_params(cls, params):
        try:
            mlflow.log_params(params)
        except ConnectionError:
            logger.warning("ConnectionError in logging params to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        try:
            mlflow.log_artifacts(dir_path, artifact_path)
        except ConnectionError:
            logger.warning(f"ConnectionError in logging artifacts to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    @classmethod
    def end_run(cls):
        mlflow.end_run()


class TensorBoardLogger(BaseMLLogger):
    """
    PyTorch TensorBoard Logger
    """

    def __init__(self, **kwargs):
        from tensorboardX import SummaryWriter
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
    contains_named_entity = len([x for x in preds if "B-" in x]) != 0
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
    if contains_named_entity and len(simple_tags) == 0:
        raise Exception("Predicted Named Entities lost when converting from IOB to simple tags. Please check the format"
                        "of the training data adheres to either adheres to IOB2 format or is converted when "
                        "read_ner_file() is called.")
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

def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(json.dumps(payload_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return checksum


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(json.dumps(payload_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return checksum

def reformat_msmarco_train(filename, output_filename):
    """
    Given a df of structure [query, pos_passage, neg_passage], this function converts it to [query, passage, label]
    """
    print("Reformatting MSMarco train data...")
    df = pd.read_csv(filename, header=None, sep="\t")
    samples = []
    for i, row in tqdm(df.iterrows()):
       query = row[0]
       pos = row[1]
       neg = row[2]
       samples.append([query, pos, 1])
       samples.append([query, neg, 0])
    with open(output_filename, "w") as f:
        f.write("text\ttext_b\tlabel\n")
        for (query, passage, label) in samples:
            f.write(f"{query}\t{passage}\t{label}\n")
    print(f"MSMarco train data saved at {output_filename}")

def reformat_msmarco_dev(queries_filename, passages_filename, qrels_filename, top1000_filename, output_filename):
    print("Reformatting MSMarco dev data...")
    top1000_file = open(top1000_filename)
    qrels_file = open(qrels_filename)
    queries_file = open(queries_filename)
    passages_file = open(passages_filename)

    # Generate a top1000 dict
    top1000 = dict()
    for l in tqdm(top1000_file):
        qid, pid, _, _ = l.split("\t")
        if qid not in top1000:
            top1000[qid] = []
        top1000[qid].append(pid)

    # Generate a qrels dict
    qrels = dict()
    for l in qrels_file:
        qid, _, pid, _ = l.split("\t")
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(pid)

    # Generate a queries dict
    queries = dict()
    for l in queries_file:
        qid, query = l.split("\t")
        queries[qid] = query[:-1]

    # Generate a passages dict
    passages = dict()
    for l in tqdm(passages_file):
        pid, passage = l.split("\t")
        passages[pid] = passage[:-1]

    # Generate dict with all needed info
    final = dict()
    for qid in tqdm(top1000):
        if qid not in final:
            final[qid] = []
        query = queries[qid]
        curr_qrel = qrels[qid]
        curr_top1000 = top1000[qid]
        for ct in curr_top1000:
            is_relevant = int(ct in curr_qrel)
            passage = passages[ct]
            quad = list([query, ct, passage, is_relevant])
            final[qid].append(quad)

    # Flatten the structure of final and convert to df
    records = []
    for k, v in tqdm(final.items()):
        for x in v:
            records.append([k] + x)
    df = pd.DataFrame(records, columns=["qid", "text", "pid", "text_b", "label"])
    df.to_csv(output_filename, sep="\t", index=None)
    print(f"MSMarco train data saved at {output_filename}")


def write_msmarco_results(results, output_filename):
    out_file = open(output_filename, "w")
    for dictionary in results:
        for pred in dictionary["predictions"]:
            if pred["label"] == "1":
                score = pred["probability"]
            elif pred["label"] == "0":
                score = 1 - pred["probability"]
            out_file.write(str(score))
            out_file.write("\n")

def stack(list_of_lists):
    n_lists_final = len(list_of_lists[0])
    ret = [list() for _ in range(n_lists_final)]
    for l in list_of_lists:
        for i, x in enumerate(l):
            ret[i] += (x)
    return ret


def try_get(keys, dictionary):
    for key in keys:
        if key in dictionary:
            ret = dictionary[key]
            if type(ret) == list:
                ret = ret[0]
            return ret
    return None