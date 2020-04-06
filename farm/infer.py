import logging
import multiprocessing as mp
import os
from functools import partial

import torch
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from transformers.configuration_auto import AutoConfig

from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.processor import Processor, InferenceProcessor, SquadProcessor, NERProcessor, TextClassificationProcessor
from farm.data_handler.utils import grouper
from farm.modeling.tokenization import Tokenizer
from farm.modeling.adaptive_model import AdaptiveModel, BaseAdaptiveModel
from farm.utils import initialize_device_settings
from farm.utils import set_all_seeds, calc_chunksize, log_ascii_workers


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
        extraction_layer=None
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
                               (sentence vector), reduce_max (sentence vector), 'per_token' (individual token vectors)
        :type extraction_strategy: str
        :param extraction_layer: number of layer from which the embeddings shall be extracted. Default: -1 (very last layer).
        :type extraction_layer: int
        :return: An instance of the Inferencer.

        """
        # Init device and distributed settings
        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)

        self.processor = processor
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.language = self.model.get_language()
        self.task_type = task_type

        if task_type == "embeddings":
            if not extraction_layer or not extraction_strategy:
                    logger.warning("Using task_type='embeddings', but couldn't find one of the args `extraction_layer` and `extraction_strategy`. "
                                   "Since FARM 0.4.2, you set both when initializing the Inferencer and then call inferencer.inference_from_dicts() instead of inferencer.extract_vectors()")
            self.model.prediction_heads = torch.nn.ModuleList([])
            self.model.language_model.extraction_layer = extraction_layer
            self.model.language_model.extraction_strategy = extraction_strategy

        # TODO add support for multiple prediction heads

        self.name = name if name != None else f"anonymous-{self.task_type}"
        self.return_class_probs = return_class_probs

        model.connect_heads_with_processor(processor.tasks, require_labels=False)
        set_all_seeds(42)

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
        extraction_strategy=None
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
        :return: An instance of the Inferencer.

        """

        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)
        name = os.path.basename(model_name_or_path)

        # a) either from local dir
        if os.path.exists(model_name_or_path):
            model = BaseAdaptiveModel.load(load_dir=model_name_or_path, device=device, strict=strict)
            if task_type == "embeddings":
                processor = InferenceProcessor.load_from_dir(model_name_or_path)
            else:
                processor = Processor.load_from_dir(model_name_or_path)

        # b) or from remote transformers model hub
        else:
            logger.info(f"Could not find `{model_name_or_path}` locally. Try to download from model hub ...")
            if not task_type:
                raise ValueError("Please specify the 'task_type' of the model you want to load from transformers. "
                                 "Valid options for arg `task_type`:"
                                 "'question_answering', 'embeddings', 'text_classification', 'ner'")

            model = AdaptiveModel.convert_from_transformers(model_name_or_path, device, task_type)
            config = AutoConfig.from_pretrained(model_name_or_path)
            tokenizer = Tokenizer.load(model_name_or_path)

            # TODO infer task_type automatically from config (if possible)
            if task_type == "question_answering":
                processor = SquadProcessor(
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    label_list=["start_token", "end_token"],
                    metric="squad",
                    data_dir="data",
                    doc_stride=doc_stride
                )
            elif task_type == "embeddings":
                processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)

            elif task_type == "text_classification":
                label_list = list(config.id2label[id] for id in range(len(config.id2label)))
                processor = TextClassificationProcessor(tokenizer=tokenizer,
                                                        max_seq_len=max_seq_len,
                                                        data_dir="data",
                                                        label_list=label_list,
                                                        label_column_name="label",
                                                        metric="acc",
                                                        quote_char='"',
                                                        )
            elif task_type == "ner":
                label_list = list(config.label2id.keys())
                processor = NERProcessor(
                    tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir="data", metric="seq_f1",
                    label_list=label_list
                )
            else:
                raise ValueError(f"`task_type` {task_type} is not supported yet. "
                                 f"Valid options for arg `task_type`: 'question_answering', "
                                 f"'embeddings', 'text_classification', 'ner'")

        return cls(
            model,
            processor,
            task_type=task_type,
            batch_size=batch_size,
            gpu=gpu,
            name=name,
            return_class_probs=return_class_probs,
            extraction_strategy=extraction_strategy,
            extraction_layer=extraction_layer
        )

    def save(self, path):
        self.model.save(path)
        self.processor.save(path)

    def inference_from_file(self, file, max_processes=128):
        """
        Run down-stream inference on samples created from an input file.
        The file should be in the same format as the ones used during training
        (e.g. squad style for QA, tsv for doc classification ...) as the same processor will be used for conversion .

        :param file: path of the input file for Inference
        :type file: str
        :param max_processes: the maximum size of `multiprocessing.Pool`. Set to value of 1 to disable multiprocessing.
                              If you want to debug the Language Model, you might need to disable multiprocessing!
        :type max_processes: int
        """
        dicts = self.processor.file_to_dicts(file)
        preds_all = self.inference_from_dicts(dicts, rest_api_schema=False, max_processes=max_processes)
        return preds_all

    def inference_from_dicts(self, dicts, rest_api_schema=False, max_processes=128, min_chunksize=4):
        """
        Runs down-stream inference on samples created from input dictionaries.
        The format of the input `dicts` depends on the task:

        QA:                    [{"qas": ["What is X?"], "context":  "Some context containing the answer"}]
        Classification / NER / embeddings:  [{"text": "Some input text"}]


        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param rest_api_schema: Whether input dicts use the format that complies with the FARM REST API.
                                Currently only used for QA to switch from squad to a more useful format in production.
                                While input is almost the same, output contains additional meta data(offset, context..)
        :type rest_api_schema: bool
        :return: dict of predictions
        :param max_processes: The maximum size of `multiprocessing.Pool`. Set to value of 1 to disable multiprocessing.
                              If you want to debug the Language Model, you might need to disable multiprocessing!
                              For very small number of dicts, time incurred in spawning processes could outweigh
                              performance boost, eg, in the case of HTTP APIs for Inference. For such cases
                              multiprocessing should be disabled.
        :param min_chunksize: minimum number of dicts to put together in one chunk and feed to one process
                              (only relevant if you do multiprocessing)
        :type min_chunksize: int
        """

        # whether to aggregate predictions across different samples (e.g. for QA on long texts)
        aggregate_preds = False
        if len(self.model.prediction_heads) > 0:
            aggregate_preds = hasattr(self.model.prediction_heads[0], "aggregate_preds")

        # Using multiprocessing
        if max_processes > 1:  # use multiprocessing if max_processes > 1
            multiprocessing_chunk_size, num_cpus_used = calc_chunksize(len(dicts), max_processes=max_processes, min_chunksize=min_chunksize)

            # Get us some workers (i.e. processes)
            p = mp.Pool(processes=num_cpus_used)
            logger.info(
                f"Got ya {num_cpus_used} parallel workers to do inference on {len(dicts)} dicts (chunksize = {multiprocessing_chunk_size})..."
            )
            log_ascii_workers(num_cpus_used, logger)

            # We group the input dicts into chunks and feed each chunk to a different process,
            # where it gets converted to a pytorch dataset
            results = p.imap(
                partial(self._create_datasets_chunkwise, processor=self.processor, rest_api_schema=rest_api_schema),
                grouper(dicts, multiprocessing_chunk_size),
                1,
            )

            # Once a process spits out a preprocessed chunk. we feed this dataset directly to the model.
            # So we don't need to wait until all preprocessing has finished before getting first predictions.
            preds_all = []
            with tqdm(total=len(dicts), desc=f"Inferencing Dicts", unit=" Dicts") as pbar:
                for dataset, tensor_names, baskets in results:
                    # TODO change format of formatted_preds in QA (list of dicts)
                    if aggregate_preds:
                        preds_all.extend(self._get_predictions_and_aggregate(dataset, tensor_names, baskets, rest_api_schema, disable_tqdm=True))
                    else:
                        preds_all.extend(self._get_predictions(dataset, tensor_names, baskets, rest_api_schema, disable_tqdm=True))
                    pbar.update(multiprocessing_chunk_size)
            p.close()
            p.join()
        # Using single process (helpful for debugging!)
        else:
            chunk = next(grouper(dicts, len(dicts)))
            dataset, tensor_names, baskets = self._create_datasets_chunkwise(chunk, processor=self.processor, rest_api_schema=rest_api_schema)
            # TODO change format of formatted_preds in QA (list of dicts)
            if aggregate_preds:
                preds_all = self._get_predictions_and_aggregate(dataset, tensor_names, baskets, rest_api_schema)
            else:
                preds_all = self._get_predictions(dataset, tensor_names, baskets, rest_api_schema)
        return preds_all

    @classmethod
    def _create_datasets_chunkwise(cls, chunk, processor, rest_api_schema):
        """Convert ONE chunk of data (i.e. dictionaries) into ONE pytorch dataset.
        This is usually executed in one of many parallel processes.
        The resulting datasets of the processes are merged together afterwards"""
        dicts = [d[1] for d in chunk]
        indices = [d[0] for d in chunk]
        dataset, tensor_names, baskets = processor.dataset_from_dicts(dicts, indices, rest_api_schema, return_baskets=True)
        return dataset, tensor_names, baskets

    def _get_predictions(self, dataset, tensor_names, baskets, rest_api_schema=False, disable_tqdm=False):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dataset: PyTorch Dataset with samples you want to predict
        :param tensor_names: Names of the tensors in the dataset
        :param baskets: For each item in the dataset, we need additional information to create formatted preds.
                        Baskets contain all relevant infos for that.
                        Example: QA - input string to convert the predicted answer from indices back to string space
        :param rest_api_schema: Whether input dicts use the format that complies with the FARM REST API.
                                Currently only used for QA to switch from squad to a more useful format in production.
                                While input is almost the same, output contains additional meta data(offset, context..)
        :type rest_api_schema: bool
        :param disable_tqdm: Whether to disable tqdm logging (can get very verbose in multiprocessing)
        :type disable_tqdm: bool
        :return: list of predictions
        """
        samples = [s for b in baskets for s in b.samples]

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        preds_all = []
        for i, batch in enumerate(tqdm(data_loader, desc=f"Inferencing Samples", unit=" Batches", disable=disable_tqdm)):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]

            # get logits
            with torch.no_grad():
                logits = self.model.forward(**batch)[0]
                preds = self.model.formatted_preds(
                    logits=[logits],
                    samples=batch_samples,
                    tokenizer=self.processor.tokenizer,
                    rest_api_schema=rest_api_schema,
                    return_class_probs=self.return_class_probs,
                    **batch)
                preds_all += preds
        return preds_all

    def _get_predictions_and_aggregate(self, dataset, tensor_names, baskets, rest_api_schema=False, disable_tqdm=False):
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
        :param rest_api_schema: Whether input dicts use the format that complies with the FARM REST API.
                                Currently only used for QA to switch from squad to a more useful format in production.
                                While input is almost the same, output contains additional meta data(offset, context..)
        :type rest_api_schema: bool
        :param disable_tqdm: Whether to disable tqdm logging (can get very verbose in multiprocessing)
        :type disable_tqdm: bool
        :return: list of predictions
        """

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        unaggregated_preds_all = []
        for i, batch in enumerate(tqdm(data_loader, desc=f"Inferencing Samples", unit=" Batches", disable=disable_tqdm)):
            batch = {key: batch[key].to(self.device) for key in batch}

            # get logits
            with torch.no_grad():
                # Aggregation works on preds, not logits. We want as much processing happening in one batch + on GPU
                # So we transform logits to preds here as well
                logits = self.model.forward(**batch)
                preds = self.model.logits_to_preds(logits, **batch)[0]
                unaggregated_preds_all += preds

        # In some use cases we want to aggregate the individual predictions.
        # This is mostly useful, if the input text is longer than the max_seq_len that the model can process.
        # In QA we can use this to get answers from long input texts by first getting predictions for smaller passages
        # and then aggregating them here.

        # can assume that we have only complete docs i.e. all the samples of one doc are in the current chunk
        preds_all = self.model.formatted_preds(logits=[None], # For QA we collected preds per batch and do not want to pass logits
                                               preds_p=unaggregated_preds_all,
                                               baskets=baskets,
                                               rest_api_schema=rest_api_schema)[0]
        return preds_all

    def extract_vectors(
        self, dicts, extraction_strategy="cls_token", extraction_layer=-1, max_processes=1
    ):
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
        :param max_processes: number of parallel processes for multiprocessing
        :type max_processes: int
        :return: dict of predictions
        """

        logger.warning("Deprecated! Please use Inferencer.inference_from_dicts() instead.")
        self.model.prediction_heads = torch.nn.ModuleList([])
        self.model.language_model.extraction_layer = extraction_layer
        self.model.language_model.extraction_strategy = extraction_strategy

        return self.inference_from_dicts(dicts, rest_api_schema=False, max_processes=max_processes)


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
