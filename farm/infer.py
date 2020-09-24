import logging
import multiprocessing as mp
import os
from functools import partial
import warnings

import torch
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from typing import Generator, List, Union

from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.processor import Processor, InferenceProcessor
from farm.data_handler.utils import grouper
from farm.data_handler.inputs import QAInput
from farm.modeling.adaptive_model import AdaptiveModel, BaseAdaptiveModel, ONNXAdaptiveModel
from farm.modeling.optimization import optimize_model
from farm.utils import initialize_device_settings
from farm.utils import set_all_seeds, calc_chunksize, log_ascii_workers, Benchmarker
from farm.modeling.predictions import QAPred

logger = logging.getLogger(__name__)


class Inferencer:
    """
    Loads a saved AdaptiveModel/ONNXAdaptiveModel from disk and runs it in inference mode. Can be used for a
    model with prediction head (down-stream predictions) and without (using LM as embedder).

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
       model = Inferencer.load(your_model_dir, extraction_strategy="cls_token", extraction_layer=-1)
       model.inference_from_dicts(dicts=basic_texts)
    """

    def __init__(
        self,
        model,
        processor,
        task_type,
        batch_size=4,
        gpu=False,
        name=None,
        return_class_probs=False,
        extraction_strategy=None,
        extraction_layer=None,
        s3e_stats=None,
        num_processes=None,
        disable_tqdm=False,
        benchmarking=False,
        dummy_ph=False
    ):
        """
        Initializes Inferencer from an AdaptiveModel and a Processor instance.

        :param model: AdaptiveModel to run in inference mode
        :type model: AdaptiveModel
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :type processor: Processor
        :param task_type: Type of task the model should be used for. Currently supporting:
                          "embeddings", "question_answering", "text_classification", "ner". More coming soon...
        :param task_type: str
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :param name: Name for the current Inferencer model, displayed in the REST API
        :type name: string
        :param return_class_probs: either return probability distribution over all labels or the prob of the associated label
        :type return_class_probs: bool
        :param extraction_strategy: Strategy to extract vectors. Choices: 'cls_token' (sentence vector), 'reduce_mean'
                               (sentence vector), reduce_max (sentence vector), 'per_token' (individual token vectors),
                               's3e' (sentence vector via S3E pooling, see https://arxiv.org/abs/2002.09620)
        :type extraction_strategy: str
        :param extraction_layer: number of layer from which the embeddings shall be extracted. Default: -1 (very last layer).
        :type extraction_layer: int
        :param s3e_stats: Stats of a fitted S3E model as returned by `fit_s3e_on_corpus()`
                          (only needed for task_type="embeddings" and extraction_strategy = "s3e")
        :type s3e_stats: dict
        :param num_processes: the number of processes for `multiprocessing.Pool`.
                              Set to value of 1 (or 0) to disable multiprocessing.
                              Set to None to let Inferencer use all CPU cores minus one.
                              If you want to debug the Language Model, you might need to disable multiprocessing!
                              **Warning!** If you use multiprocessing you have to close the
                              `multiprocessing.Pool` again! To do so call
                              :func:`~farm.infer.Inferencer.close_multiprocessing_pool` after you are
                              done using this class. The garbage collector will not do this for you!
        :type num_processes: int
        :param disable_tqdm: Whether to disable tqdm logging (can get very verbose in multiprocessing)
        :type disable_tqdm: bool
        :param dummy_ph: If True, methods of the prediction head will be replaced
                     with a dummy method. This is used to isolate lm run time from ph run time.
        :type dummy_ph: bool
        :param benchmarking: If True, a benchmarking object will be initialised within the class and
                             certain parts of the code will be timed for benchmarking. Should be kept
                             False if not benchmarking since these timing checkpoints require synchronization
                             of the asynchronous Pytorch operations and may slow down the model.
        :type benchmarking: bool
        :return: An instance of the Inferencer.

        """
        # For benchmarking
        if dummy_ph:
            model.bypass_ph()

        self.benchmarking = benchmarking
        if self.benchmarking:
            self.benchmarker = Benchmarker()

        # Init device and distributed settings
        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)

        self.processor = processor
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.language = self.model.get_language()
        self.task_type = task_type
        self.disable_tqdm = disable_tqdm

        if task_type == "embeddings":
            if not extraction_layer or not extraction_strategy:
                    logger.warning("Using task_type='embeddings', but couldn't find one of the args `extraction_layer` and `extraction_strategy`. "
                                   "Since FARM 0.4.2, you set both when initializing the Inferencer and then call inferencer.inference_from_dicts() instead of inferencer.extract_vectors()")
            self.model.prediction_heads = torch.nn.ModuleList([])
            self.model.language_model.extraction_layer = extraction_layer
            self.model.language_model.extraction_strategy = extraction_strategy
            self.model.language_model.s3e_stats = s3e_stats

        # TODO add support for multiple prediction heads

        self.name = name if name != None else f"anonymous-{self.task_type}"
        self.return_class_probs = return_class_probs

        model.connect_heads_with_processor(processor.tasks, require_labels=False)
        set_all_seeds(42)

        self._set_multiprocessing_pool(num_processes)

    @classmethod
    def load(
        cls,
        model_name_or_path,
        batch_size=4,
        gpu=False,
        task_type=None,
        return_class_probs=False,
        strict=True,
        max_seq_len=256,
        doc_stride=128,
        extraction_layer=None,
        extraction_strategy=None,
        s3e_stats=None,
        num_processes=None,
        disable_tqdm=False,
        tokenizer_class=None,
        use_fast=False,
        tokenizer_args=None,
        dummy_ph=False,
        benchmarking=False,

    ):
        """
        Load an Inferencer incl. all relevant components (model, tokenizer, processor ...) either by

        1. specifying a public name from transformers' model hub (https://huggingface.co/models)
        2. or pointing to a local directory it is saved in.

        :param model_name_or_path: Local directory or public name of the model to load.
        :type model_name_or_path: str
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :param task_type: Type of task the model should be used for. Currently supporting:
                          "embeddings", "question_answering", "text_classification", "ner". More coming soon...
        :param task_type: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :param max_seq_len: maximum length of one text sample
        :type max_seq_len: int
        :param doc_stride: Only QA: When input text is longer than max_seq_len it gets split into parts, strided by doc_stride
        :type doc_stride: int
        :param extraction_strategy: Strategy to extract vectors. Choices: 'cls_token' (sentence vector), 'reduce_mean'
                               (sentence vector), reduce_max (sentence vector), 'per_token' (individual token vectors)
        :type extraction_strategy: str
        :param extraction_layer: number of layer from which the embeddings shall be extracted. Default: -1 (very last layer).
        :type extraction_layer: int
        :param s3e_stats: Stats of a fitted S3E model as returned by `fit_s3e_on_corpus()`
                          (only needed for task_type="embeddings" and extraction_strategy = "s3e")
        :type s3e_stats: dict
        :param num_processes: the number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer use all CPU cores minus one. If you want to
                              debug the Language Model, you might need to disable multiprocessing!
                              **Warning!** If you use multiprocessing you have to close the
                              `multiprocessing.Pool` again! To do so call
                              :func:`~farm.infer.Inferencer.close_multiprocessing_pool` after you are
                              done using this class. The garbage collector will not do this for you!
        :type num_processes: int
        :param disable_tqdm: Whether to disable tqdm logging (can get very verbose in multiprocessing)
        :type disable_tqdm: bool
        :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
        :type tokenizer_class: str
        :param use_fast: (Optional, False by default) Indicate if FARM should try to load the fast version of the tokenizer (True) or
            use the Python one (False).
        :param tokenizer_args: (Optional) Will be passed to the Tokenizer ``__init__`` method.
            See https://huggingface.co/transformers/main_classes/tokenizer.html and detailed tokenizer documentation
            on `Hugging Face Transformers <https://huggingface.co/transformers/>`_.
        :type tokenizer_args: dict
        :type use_fast: bool
        :param dummy_ph: If True, methods of the prediction head will be replaced
                             with a dummy method. This is used to isolate lm run time from ph run time.
        :type dummy_ph: bool
        :param benchmarking: If True, a benchmarking object will be initialised within the class and
                             certain parts of the code will be timed for benchmarking. Should be kept
                             False if not benchmarking since these timing checkpoints require synchronization
                             of the asynchronous Pytorch operations and may slow down the model.
        :type benchmarking: bool
        :return: An instance of the Inferencer.

        """
        if tokenizer_args is None:
            tokenizer_args = {}

        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)
        name = os.path.basename(model_name_or_path)

        # a) either from local dir
        if os.path.exists(model_name_or_path):
            model = BaseAdaptiveModel.load(load_dir=model_name_or_path, device=device, strict=strict)
            if task_type == "embeddings":
                processor = InferenceProcessor.load_from_dir(model_name_or_path)
            else:
                processor = Processor.load_from_dir(model_name_or_path)

            # override processor attributes loaded from config file with inferencer params
            processor.max_seq_len = max_seq_len
            if hasattr(processor, "doc_stride"):
                assert doc_stride < max_seq_len, "doc_stride is longer than max_seq_len. This means that there will be gaps " \
                                                 "as the passage windows slide, causing the model to skip over parts of the document. "\
                                                 "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384) "
                processor.doc_stride = doc_stride

        # b) or from remote transformers model hub
        else:
            logger.info(f"Could not find `{model_name_or_path}` locally. Try to download from model hub ...")
            if not task_type:
                raise ValueError("Please specify the 'task_type' of the model you want to load from transformers. "
                                 "Valid options for arg `task_type`:"
                                 "'question_answering', 'embeddings', 'text_classification', 'ner'")

            model = AdaptiveModel.convert_from_transformers(model_name_or_path, device, task_type)
            processor = Processor.convert_from_transformers(model_name_or_path, task_type, max_seq_len, doc_stride,
                                                            tokenizer_class, tokenizer_args, use_fast)

        if not isinstance(model,ONNXAdaptiveModel):
            model, _ = optimize_model(model=model, device=device, local_rank=-1, optimizer=None)
        return cls(
            model,
            processor,
            task_type=task_type,
            batch_size=batch_size,
            gpu=gpu,
            name=name,
            return_class_probs=return_class_probs,
            extraction_strategy=extraction_strategy,
            extraction_layer=extraction_layer,
            s3e_stats=s3e_stats,
            num_processes=num_processes,
            disable_tqdm=disable_tqdm,
            benchmarking=benchmarking,
            dummy_ph=dummy_ph
        )

    def _set_multiprocessing_pool(self, num_processes):
        """
        Initialize a multiprocessing.Pool for instances of Inferencer.

        :param num_processes: the number of processes for `multiprocessing.Pool`.
                              Set to value of 1 (or 0) to disable multiprocessing.
                              Set to None to let Inferencer use all CPU cores minus one.
                              If you want to debug the Language Model, you might need to disable multiprocessing!
                              **Warning!** If you use multiprocessing you have to close the
                              `multiprocessing.Pool` again! To do so call
                              :func:`~farm.infer.Inferencer.close_multiprocessing_pool` after you are
                              done using this class. The garbage collector will not do this for you!
        :type num_processes: int
        :return:
        """
        self.process_pool = None
        if num_processes == 0 or num_processes == 1:  # disable multiprocessing
            self.process_pool = None
        else:
            if num_processes is None:  # use all CPU cores
                num_processes = mp.cpu_count() - 1
            self.process_pool = mp.Pool(processes=num_processes)
            logger.info(
                f"Got ya {num_processes} parallel workers to do inference ..."
            )
            log_ascii_workers(n=num_processes,logger=logger)

    def close_multiprocessing_pool(self, join=False):
        """Close the `multiprocessing.Pool` again.

        If you use multiprocessing you have to close the `multiprocessing.Pool` again!
        To do so call this function after you are done using this class.
        The garbage collector will not do this for you!

        :param join: wait for the worker processes to exit
        :type join: bool
        """
        if self.process_pool is not None:
            self.process_pool.close()
            if join:
                self.process_pool.join()
            self.process_pool = None

    def save(self, path):
        self.model.save(path)
        self.processor.save(path)

    def inference_from_file(self, file, multiprocessing_chunksize=None, streaming=False, return_json=True):
        """
        Run down-stream inference on samples created from an input file.
        The file should be in the same format as the ones used during training
        (e.g. squad style for QA, tsv for doc classification ...) as the same Processor will be used for conversion.

        :param file: path of the input file for Inference
        :type file: str
        :param multiprocessing_chunksize: number of dicts to put together in one chunk and feed to one process
        :type multiprocessing_chunksize: int
        :param streaming: return a Python generator object that yield results as they get computed, instead of
                          blocking for all the results. To use streaming, the dicts parameter must be a generator
                          and num_processes argument must be set. This mode can be useful to implement large scale
                          non-blocking inference pipelines.
        :type streaming: bool

        :return: an iterator(list or generator) of predictions
        :rtype: iter
        """
        dicts = self.processor.file_to_dicts(file)
        preds_all = self.inference_from_dicts(
            dicts,
            return_json=return_json,
            multiprocessing_chunksize=multiprocessing_chunksize,
            streaming=streaming,
        )
        if streaming:
            return preds_all
        else:
            return list(preds_all)

    def inference_from_dicts(
        self, dicts, return_json=True, multiprocessing_chunksize=None, streaming=False
    ):
        """
        Runs down-stream inference on samples created from input dictionaries.
        The format of the input `dicts` depends on the task:

        * QA (SQuAD style):    [{"qas": ["What is X?"], "context":  "Some context containing the answer"}] (Deprecated)
        * QA (FARM style): [{"questions": ["What is X?"], "text":  "Some context containing the answer"}]
        * Classification / NER / embeddings: [{"text": "Some input text"}]

        Inferencer has a high performance non-blocking streaming mode for large scale inference use cases. With this
        mode, the dicts parameter can optionally be a Python generator object that yield dicts, thus avoiding loading
        dicts in memory. The inference_from_dicts() method returns a generator that yield predictions. To use streaming,
        set the streaming param to True and determine optimal multiprocessing_chunksize by performing speed benchmarks.


        :param dicts: Samples to run inference on provided as a list(or a generator object) of dicts.
                      One dict per sample.
        :type dicts: iter(dict)
        :param return_json: Whether the output should be in a json appropriate format. If False, it returns the prediction
                            object where applicable, else it returns PredObj.to_json()
        :type return_json: bool
        :return: dict of predictions
        :param multiprocessing_chunksize: number of dicts to put together in one chunk and feed to one process
                                          (only relevant if you do multiprocessing)
        :type multiprocessing_chunksize: int
        :param streaming: return a Python generator object that yield results as they get computed, instead of blocking
                          for all the results. To use streaming, the dicts parameter must be a generator and
                          num_processes argument must be set. This mode can be useful to implement large scale
                          non-blocking inference pipelines.
        :type streaming: bool

        :return: an iterator(list or generator) of predictions
        :rtype: iter
        """

        # whether to aggregate predictions across different samples (e.g. for QA on long texts)
        if set(dicts[0].keys()) == {"qas", "context"}:
            warnings.warn("QA Input dictionaries with [qas, context] as keys will be deprecated in the future",
                          DeprecationWarning)

        aggregate_preds = False
        if len(self.model.prediction_heads) > 0:
            aggregate_preds = hasattr(self.model.prediction_heads[0], "aggregate_preds")

        if self.process_pool is None:  # multiprocessing disabled (helpful for debugging or using in web frameworks)
            predictions = self._inference_without_multiprocessing(dicts, return_json, aggregate_preds)
            return predictions
        else:  # use multiprocessing for inference
            # Calculate values of multiprocessing_chunksize and num_processes if not supplied in the parameters.
            # The calculation of the values is based on whether streaming mode is enabled. This is only for speed
            # optimization and do not impact the results of inference.
            if streaming:
                if multiprocessing_chunksize is None:
                    logger.warning("Streaming mode is enabled for the Inferencer but multiprocessing_chunksize is not "
                                   "supplied. Continuing with a default value of 20. Perform benchmarking on your data "
                                   "to get the optimal chunksize.")
                    multiprocessing_chunksize = 20
            else:
                if multiprocessing_chunksize is None:
                    _chunk_size, _ = calc_chunksize(len(dicts))
                    multiprocessing_chunksize = _chunk_size

            predictions = self._inference_with_multiprocessing(
                dicts, return_json, aggregate_preds, multiprocessing_chunksize,
            )

            # return a generator object if streaming is enabled, else, cast the generator to a list.
            if not streaming and type(predictions) != list:
                return list(predictions)
            else:
                return predictions

    def _inference_without_multiprocessing(self, dicts, return_json, aggregate_preds):
        """
        Implementation of inference from dicts without using Python multiprocessing. Useful for debugging or in API
        framework where spawning new processes could be expensive.

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: iter(dict)
        :param return_json: Whether the output should be in a json appropriate format. If False, it returns the prediction
                            object where applicable, else it returns PredObj.to_json()
        :type return_json: bool
        :param aggregate_preds: whether to aggregate predictions across different samples (e.g. for QA on long texts)
        :type aggregate_preds: bool

        :return: list of predictions
        :rtype: list
        """
        dataset, tensor_names, baskets = self.processor.dataset_from_dicts(
            dicts, indices=[i for i in range(len(dicts))], return_baskets=True
        )

        if self.benchmarking:
            self.benchmarker.record("dataset_single_proc")

        # TODO change format of formatted_preds in QA (list of dicts)
        if aggregate_preds:
            preds_all = self._get_predictions_and_aggregate(dataset, tensor_names, baskets)
        else:
            preds_all = self._get_predictions(dataset, tensor_names, baskets)

        if return_json:
            # TODO this try catch should be removed when all tasks return prediction objects
            try:
                preds_all = [x.to_json() for x in preds_all]
            except AttributeError:
                pass

        return preds_all

    def _inference_with_multiprocessing(
        self, dicts, return_json, aggregate_preds, multiprocessing_chunksize
    ):
        """
        Implementation of inference. This method is a generator that yields the results.

        :param dicts: Samples to run inference on provided as a list of dicts or a generator object that yield dicts.
        :type dicts: iter(dict)
        :param return_json: Whether the output should be in a json appropriate format. If False, it returns the prediction
                            object where applicable, else it returns PredObj.to_json()
        :type return_json: bool
        :param aggregate_preds: whether to aggregate predictions across different samples (e.g. for QA on long texts)
        :type aggregate_preds: bool
        :param multiprocessing_chunksize: number of dicts to put together in one chunk and feed to one process
        :type multiprocessing_chunksize: int
        :return: generator object that yield predictions
        :rtype: iter
        """

        # We group the input dicts into chunks and feed each chunk to a different process
        # in the pool, where it gets converted to a pytorch dataset
        results = self.process_pool.imap(
            partial(self._create_datasets_chunkwise, processor=self.processor),
            grouper(iterable=dicts, n=multiprocessing_chunksize),
            1,
        )

        # Once a process spits out a preprocessed chunk. we feed this dataset directly to the model.
        # So we don't need to wait until all preprocessing has finished before getting first predictions.
        for dataset, tensor_names, baskets in results:
            # TODO change format of formatted_preds in QA (list of dicts)
            if aggregate_preds:
                predictions = self._get_predictions_and_aggregate(
                    dataset, tensor_names, baskets
                )
            else:
                predictions = self._get_predictions(dataset, tensor_names, baskets)

            if return_json:
                # TODO this try catch should be removed when all tasks return prediction objects
                try:
                    predictions = [x.to_json() for x in predictions]
                except AttributeError:
                    pass
            yield from predictions

    @classmethod
    def _create_datasets_chunkwise(cls, chunk, processor):
        """Convert ONE chunk of data (i.e. dictionaries) into ONE pytorch dataset.
        This is usually executed in one of many parallel processes.
        The resulting datasets of the processes are merged together afterwards"""
        dicts = [d[1] for d in chunk]
        indices = [d[0] for d in chunk]
        dataset, tensor_names, baskets = processor.dataset_from_dicts(dicts, indices, return_baskets=True)
        return dataset, tensor_names, baskets

    def _get_predictions(self, dataset, tensor_names, baskets):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dataset: PyTorch Dataset with samples you want to predict
        :param tensor_names: Names of the tensors in the dataset
        :param baskets: For each item in the dataset, we need additional information to create formatted preds.
                        Baskets contain all relevant infos for that.
                        Example: QA - input string to convert the predicted answer from indices back to string space
        :return: list of predictions
        """
        samples = [s for b in baskets for s in b.samples]

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        preds_all = []
        for i, batch in enumerate(tqdm(data_loader, desc=f"Inferencing Samples", unit=" Batches", disable=self.disable_tqdm)):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]

            # get logits
            with torch.no_grad():
                logits = self.model.forward(**batch)[0]
                preds = self.model.formatted_preds(
                    logits=[logits],
                    samples=batch_samples,
                    tokenizer=self.processor.tokenizer,
                    return_class_probs=self.return_class_probs,
                    **batch)
                preds_all += preds
        return preds_all

    def _get_predictions_and_aggregate(self, dataset, tensor_names, baskets):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + logits_to_preds + formatted_preds).

        Difference to _get_predictions():
         - Additional aggregation step across predictions of individual samples
         (e.g. For QA on long texts, we extract answers from multiple passages and then aggregate them on the "document level")

        :param dataset: PyTorch Dataset with samples you want to predict
        :param tensor_names: Names of the tensors in the dataset
        :param baskets: For each item in the dataset, we need additional information to create formatted preds.
                        Baskets contain all relevant infos for that.
                        Example: QA - input string to convert the predicted answer from indices back to string space
        :return: list of predictions
        """

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        # TODO Sometimes this is the preds of one head, sometimes of two. We need a more advanced stacking operation
        # TODO so that preds of the right shape are passed in to formatted_preds
        unaggregated_preds_all = []

        for i, batch in enumerate(tqdm(data_loader, desc=f"Inferencing Samples", unit=" Batches", disable=self.disable_tqdm)):

            batch = {key: batch[key].to(self.device) for key in batch}

            # get logits
            with torch.no_grad():
                # Aggregation works on preds, not logits. We want as much processing happening in one batch + on GPU
                # So we transform logits to preds here as well
                logits = self.model.forward(**batch)
                # preds = self.model.logits_to_preds(logits, **batch)[0] (This must somehow be useful for SQuAD)
                preds = self.model.logits_to_preds(logits, **batch)
                unaggregated_preds_all.append(preds)

        # In some use cases we want to aggregate the individual predictions.
        # This is mostly useful, if the input text is longer than the max_seq_len that the model can process.
        # In QA we can use this to get answers from long input texts by first getting predictions for smaller passages
        # and then aggregating them here.

        # At this point unaggregated preds has shape [n_batches][n_heads][n_samples]

        # can assume that we have only complete docs i.e. all the samples of one doc are in the current chunk
        logits = [None]
        preds_all = self.model.formatted_preds(logits=logits, # For QA we collected preds per batch and do not want to pass logits
                                               preds=unaggregated_preds_all,
                                               baskets=baskets)
        if self.benchmarking:
            self.benchmarker.record("formatted_preds")
        return preds_all

    def extract_vectors(self, dicts, extraction_strategy="cls_token", extraction_layer=-1):
        """
        Converts a text into vector(s) using the language model only (no prediction head involved).

        Example:
            basic_texts = [{"text": "Some text we want to embed"}, {"text": "And a second one"}]
            result = inferencer.extract_vectors(dicts=basic_texts)

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param extraction_strategy: Strategy to extract vectors. Choices: 'cls_token' (sentence vector), 'reduce_mean'
                               (sentence vector), reduce_max (sentence vector), 'per_token' (individual token vectors)
        :type extraction_strategy: str
        :param extraction_layer: number of layer from which the embeddings shall be extracted. Default: -1 (very last layer).
        :type extraction_layer: int
        :return: dict of predictions
        """

        logger.warning("Deprecated! Please use Inferencer.inference_from_dicts() instead.")
        self.model.prediction_heads = torch.nn.ModuleList([])
        self.model.language_model.extraction_layer = extraction_layer
        self.model.language_model.extraction_strategy = extraction_strategy

        return self.inference_from_dicts(dicts)


class QAInferencer(Inferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_type != "question_answering":
            logger.warning("QAInferencer always has task_type='question_answering' even if another value is provided "
                           "to Inferencer.load() or QAInferencer()")
            self.task_type = "question_answering"

    def inference_from_dicts(self,
                             dicts,
                             return_json=True,
                             multiprocessing_chunksize=None,
                             streaming=False) -> Union[List[QAPred], Generator[QAPred, None, None]]:
        return Inferencer.inference_from_dicts(self, dicts, return_json=return_json,
                                               multiprocessing_chunksize=multiprocessing_chunksize, streaming=streaming)

    def inference_from_file(self,
                            file,
                            multiprocessing_chunksize=None,
                            streaming=False,
                            return_json=True) -> Union[List[QAPred], Generator[QAPred, None, None]]:
        return Inferencer.inference_from_file(self, file, return_json=return_json,
                                              multiprocessing_chunksize=multiprocessing_chunksize, streaming=streaming)

    def inference_from_objects(self,
                               objects: List[QAInput],
                               return_json=True,
                               multiprocessing_chunksize=None,
                               streaming=False) -> Union[List[QAPred], Generator[QAPred, None, None]]:
        dicts = [o.to_dict() for o in objects]
        return self.inference_from_dicts(dicts, return_json=return_json,
                                         multiprocessing_chunksize=multiprocessing_chunksize, streaming=streaming)


class FasttextInferencer:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name if name != None else f"anonymous-fasttext"
        self.prediction_type = "embedder"

    @classmethod
    def load(cls, load_dir, batch_size=4, gpu=False):
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
