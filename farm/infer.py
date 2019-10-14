import os
import torch
import logging
import multiprocessing as mp
import numpy as np
from contextlib import ExitStack
from functools import partial

from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from farm.data_handler.dataloader import NamedDataLoader
from farm.modeling.adaptive_model import AdaptiveModel

from farm.utils import initialize_device_settings
from farm.data_handler.processor import Processor, InferenceProcessor
from farm.utils import set_all_seeds
from farm.utils import log_ascii_workers
from farm.data_handler.utils import grouper


logger = logging.getLogger(__name__)


class Inferencer:
    """
    Loads a saved AdaptiveModel from disk and runs it in inference mode. Can be used for a model with prediction head (down-stream predictions) and without (using LM as embedder).

    Example usage:

    .. code-block:: python

       # down-stream inference
       basic_texts = [
           {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
           {"text": "Martin Müller spielt Handball in Berlin"},
       ]
       model = Inferencer.load(your_model_dir)
       model.inference_from_dicts(dicts=basic_texts)
       # LM embeddings
       model.extract_vectors(dicts=basic_texts)

    """

    def __init__(
        self,
        model,
        processor,
        batch_size=4,
        gpu=False,
        name=None,
        return_class_probs=False
    ):
        """
        Initializes Inferencer from an AdaptiveModel and a Processor instance.

        :param model: AdaptiveModel to run in inference mode
        :type model: AdaptiveModel
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :type processor: Processor
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :param name: Name for the current Inferencer model, displayed in the REST API
        :type name: string
        :param return_class_probs: either return probability distribution over all labels or the prob of the associated label
        :type return_class_probs: bool
        :return: An instance of the Inferencer.

        """
        # Init device and distributed settings
        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, fp16=False)

        self.processor = processor
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.language = self.model.language_model.language
        # TODO adjust for multiple prediction heads
        if len(self.model.prediction_heads) == 1:
            self.prediction_type = self.model.prediction_heads[0].model_type
            # self.label_map = self.processor.label_maps[0]
        elif len(self.model.prediction_heads) == 0:
            self.prediction_type = "embedder"
        # else:
        #     raise NotImplementedError("A model with multiple prediction heads is currently not supported by the Inferencer")
        self.name = name if name != None else f"anonymous-{self.prediction_type}"
        self.return_class_probs = return_class_probs

        model.connect_heads_with_processor(processor.tasks, require_labels=False)
        set_all_seeds(42, n_gpu)

    @classmethod
    def load(
        cls,
        load_dir,
        batch_size=4,
        gpu=False,
        embedder_only=False,
        return_class_probs=False,
    ):
        """
        Initializes Inferencer from directory with saved model.

        :param load_dir: Directory where the saved model is located.
        :type load_dir: str
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :param embedder_only: If true, a faster processor (InferenceProcessor) is loaded. This should only be used
        for extracting embeddings (no downstream predictions).
        :type embedder_only: bool
        :return: An instance of the Inferencer.
        """

        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, fp16=False)

        model = AdaptiveModel.load(load_dir, device)
        if embedder_only:
            # model.prediction_heads = []
            processor = InferenceProcessor.load_from_dir(load_dir)
        else:
            processor = Processor.load_from_dir(load_dir)

        name = os.path.basename(load_dir)
        return cls(
            model,
            processor,
            batch_size=batch_size,
            gpu=gpu,
            name=name,
            return_class_probs=return_class_probs,
        )

    def inference_from_file(self, file):
        dicts = self.processor.file_to_dicts(file)
        preds_all = self.inference_from_dicts(dicts, rest_api_schema=False)
        return preds_all

    def inference_from_dicts(self, dicts, rest_api_schema=False):
        """
        Runs down-stream inference using the prediction head.

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param rest_api_schema: whether conform to the schema used for dicts in the HTTP API for Inference.
        :type rest_api_schema: bool
        :return: dict of predictions

        """
        if self.prediction_type == "embedder":
            raise TypeError(
                "You have called inference_from_dicts for a model without any prediction head! "
                "If you want to: "
                "a) ... extract vectors from the language model: call `Inferencer.extract_vectors(...)`"
                f"b) ... run inference on a downstream task: make sure your model path {self.name} contains a saved prediction head"
            )

        num_cpus = mp.cpu_count() or 1
        dicts_per_cpu = np.ceil(len(dicts) / num_cpus)
        # automatic adjustment of multiprocessing chunksize
        # for small files (containing few dicts) we want small chunksize to ulitize all available cores but never less
        # than 2, because we need it to sample another random sentence in LM finetuning
        # for large files we want to minimize processor spawning without giving too much data to one process, so we
        # clip it at 5k
        multiprocessing_chunk_size = int(np.clip((np.ceil(dicts_per_cpu / 5)), a_min=2, a_max=5000))
        dict_batches_to_process = int(len(dicts) / multiprocessing_chunk_size)
        num_cpus_used = min(mp.cpu_count(), dict_batches_to_process) or 1

        with ExitStack() as stack:
            p = stack.enter_context(mp.Pool(processes=num_cpus_used))

            logger.info(
                f"Got ya {num_cpus_used} parallel workers to do inference on {len(dicts)}dicts (chunksize = {multiprocessing_chunk_size})..."
            )
            log_ascii_workers(num_cpus_used, logger)

            results = p.imap(
                partial(self._multiproc, processor=self.processor, rest_api_schema=rest_api_schema),
                grouper(dicts, multiprocessing_chunk_size),
                1,
            )

            preds_all = []
            with tqdm(total=len(dicts), unit=" Dicts") as pbar:
                for dataset, tensor_names, sample in results:
                    preds_all.extend(self._run_inference(dataset, tensor_names, sample))
                    pbar.update(multiprocessing_chunk_size)

        return preds_all

    @classmethod
    def _multiproc(cls, chunk, processor, rest_api_schema):
        dicts = [d[1] for d in chunk]
        dataset, tensor_names = processor.dataset_from_dicts(dicts, rest_api_schema)
        samples = []
        for d in dicts:
            samples.extend(processor._dict_to_samples(d))

        return dataset, tensor_names, samples

    def _run_inference(self, dataset, tensor_names, samples):
        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )

        preds_all = []
        for i, batch in enumerate(data_loader):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]
            with torch.no_grad():
                logits = self.model.forward(**batch)
                preds = self.model.formatted_preds(
                    logits=logits,
                    samples=batch_samples,  # TODO batch_samples and logits are not aligned
                    tokenizer=self.processor.tokenizer,
                    return_class_probs=self.return_class_probs,
                    **batch,
                )
                preds_all += preds

        return preds_all

    def extract_vectors(self, dicts, extraction_strategy="cls_token", extraction_layer=-1):
        """
        Converts a text into vector(s) using the language model only (no prediction head involved).

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param extraction_strategy: Strategy to extract vectors. Choices: 'cls_token' (sentence vector), 'reduce_mean'
                               (sentence vector), reduce_max (sentence vector), 'per_token' (individual token vectors)
        :type extraction_strategy: str
        :param extraction_layer: number of layer from which the embeddings shall be extracted. Default: -1 (very last layer).
        :type: int
        :return: dict of predictions
        """

        dataset, tensor_names = self.processor.dataset_from_dicts(dicts, rest_api_schema=True)
        samples = []
        for dict in dicts:
            samples.extend(self.processor._dict_to_samples(dict))

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )

        preds_all = []
        for i, batch in enumerate(data_loader):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]
            with torch.no_grad():
                preds = self.model.language_model.formatted_preds(
                    extraction_strategy=extraction_strategy,
                    samples=batch_samples,
                    tokenizer=self.processor.tokenizer,
                    extraction_layer=extraction_layer,
                    **batch,
                )
                preds_all += preds

        return preds_all


class FasttextInferencer:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name if name != None else f"anonymous-fasttext"
        self.prediction_type = "embedder"

    @classmethod
    def load(cls, load_dir, batch_size=4, gpu=False, embedder_only=True):
        import fasttext

        if os.path.isfile(load_dir):
            return cls(model=fasttext.load_model(load_dir))
        else:
            logger.error(f"Fasttext model file does not exist at: {load_dir}")

    def extract_vectors(self, dicts, extraction_strategy="reduce_mean"):
        """
        Converts a text into vector(s) using the language model only (no prediction head involved).

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param extraction_strategy: Strategy to extract vectors. Choices: 'reduce_mean' (mean sentence vector), 'reduce_max' (max per embedding dim), 'CLS'
        :type extraction_strategy: str
        :return: dict of predictions
        """

        preds_all = []
        for d in dicts:
            pred = {}
            pred["context"] = d["text"]
            if extraction_strategy == "reduce_mean":
                pred["vec"] = self.model.get_sentence_vector(d["text"])
            else:
                raise NotImplementedError
            preds_all.append(pred)

        return preds_all
