import logging
import multiprocessing as mp
import os
from contextlib import ExitStack
from functools import partial

import torch
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from transformers.configuration_auto import AutoConfig

from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.processor import Processor, InferenceProcessor, SquadProcessor, NERProcessor, TextClassificationProcessor
from farm.data_handler.utils import grouper
from farm.modeling.tokenization import Tokenizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.utils import initialize_device_settings
from farm.utils import set_all_seeds, calc_chunksize, log_ascii_workers


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
        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)

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
        max_seq_len=256
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
                          "embeddings", "question_answering", "text_classification". More coming soon...
        :param task_type: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :return: An instance of the Inferencer.

        """

        device, n_gpu = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)
        name = os.path.basename(model_name_or_path)

        # a) either from local dir
        if os.path.exists(model_name_or_path):
            model = AdaptiveModel.load(model_name_or_path, device, strict=strict)
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
                                 "'question_answering', 'embeddings', 'text_classification'")

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
                    data_dir=None,
                )
            elif task_type == "embeddings":
                processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)

            elif task_type == "text_classification":
                label_list = list(config.id2label[id] for id in range(len(config.id2label)))
                processor = TextClassificationProcessor(tokenizer=tokenizer,
                                                        max_seq_len=max_seq_len,
                                                        data_dir=None,
                                                        label_list=label_list,
                                                        label_column_name="label",
                                                        metric="acc",
                                                        quote_char='"',
                                                        )

            # elif task_type == "multilabel-classification":
            #     # label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            #     label_list = list(config.label2id.keys())
            #
            #     processor = TextClassificationProcessor(tokenizer=tokenizer,
            #                                             max_seq_len=max_seq_len,
            #                                             data_dir=None,
            #                                             label_list=label_list,
            #                                             label_column_name="label",
            #                                             metric="acc",
            #                                             quote_char='"',
            #                                             multilabel=True,
            #                                             )

            elif task_type == "ner":
                label_list = list(config.label2id.keys())
                processor = NERProcessor(
                    tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir=None, metric="seq_f1",
                    label_list=label_list
                )
            else:
                raise ValueError(f"`task_type` {task_type} is not supported yet. "
                                 f"Valid options for arg `task_type`: 'question_answering', 'embeddings', 'text_classification'")

        return cls(
            model,
            processor,
            batch_size=batch_size,
            gpu=gpu,
            name=name,
            return_class_probs=return_class_probs,
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
        :type max_processes: int
        """
        dicts = self.processor.file_to_dicts(file)
        preds_all = self.inference_from_dicts(dicts, rest_api_schema=False, max_processes=max_processes)
        return preds_all

    def inference_from_dicts(self, dicts, rest_api_schema=False, max_processes=128):
        """
        Runs down-stream inference on samples created from input dictionaries.
        The format of the input `dicts` depends on the task:

        QA:                    [{"qas": ["What is X?"], "context":  "Some context containing the answer"}]
        Classification / NER:  [{"text": "Some input text"}]


        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param rest_api_schema: Whether input dicts use the format that complies with the FARM REST API.
                                Currently only used for QA to switch from squad to a more useful format in production.
                                While input is almost the same, output contains additional meta data(offset, context..)
        :type rest_api_schema: bool
        :return: dict of predictions
        :param max_processes: the maximum size of `multiprocessing.Pool`. Set to value of 1 to disable multiprocessing.
            For very small number of dicts, time incurred in spawning processes could outweigh performance boost, eg,
            in the case of HTTP APIs for Inference. For such cases, multiprocessing can be disabled using this param.
        """
        if self.prediction_type == "embedder":
            raise TypeError(
                "You have called inference_from_dicts for a model without any prediction head! "
                "If you want to: "
                "a) ... extract vectors from the language model: call `Inferencer.extract_vectors(...)`"
                f"b) ... run inference on a downstream task: make sure your model path {self.name} contains a saved prediction head"
            )

        # Using multiprocessing
        if max_processes > 1:  # use multiprocessing if max_processes > 1
            multiprocessing_chunk_size, num_cpus_used = calc_chunksize(len(dicts), max_processes)
            with ExitStack() as stack:

                # Get us some workers (i.e. processes)
                p = stack.enter_context(mp.Pool(processes=num_cpus_used))
                logger.info(
                    f"Got ya {num_cpus_used} parallel workers to do inference on {len(dicts)}dicts (chunksize = {multiprocessing_chunk_size})..."
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
                with tqdm(total=len(dicts), unit=" Dicts") as pbar:
                    for dataset, tensor_names, baskets in results:
                        # TODO change formot of formatted_preds in QA (list of dicts)
                        preds_all.extend(self._get_predictions(dataset, tensor_names, baskets, rest_api_schema))
                        pbar.update(multiprocessing_chunk_size)

        # Using single process (helpful for debugging!)
        else:
            chunk = next(grouper(dicts, len(dicts)))
            dataset, tensor_names, baskets = self._create_datasets_chunkwise(chunk, processor=self.processor, rest_api_schema=rest_api_schema)
            # TODO change formot of formatted_preds in QA (list of dicts)
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

    def _get_predictions(self, dataset, tensor_names, baskets, rest_api_schema=False):
        """ Feed the preprocessed dataset to the model and get the actual predictions"""
        samples = [s for b in baskets for s in b.samples]

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        logits_all = []
        preds_all = []
        aggregate_preds = hasattr(self.model.prediction_heads[0], "aggregate_preds")
        for i, batch in enumerate(tqdm(data_loader, desc=f"Inferencing")):
            batch = {key: batch[key].to(self.device) for key in batch}
            if not aggregate_preds:
                batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]

            # get logits
            with torch.no_grad():
                logits = self.model.forward(**batch)[0]

                # either just stack the logits (and convert later to readable predictions)
                if aggregate_preds:
                    logits_all += [l for l in logits]

                # or convert directly
                else:
                    preds = self.model.formatted_preds(
                        logits=[logits],
                        samples=batch_samples,
                        tokenizer=self.processor.tokenizer,
                        return_class_probs=self.return_class_probs,
                        rest_api_schema=rest_api_schema,
                        **batch)
                    preds_all += preds

        # In some use cases we want to aggregate the individual predictions.
        # This is mostly useful, if the input text is longer than the max_seq_len that the model can process.
        # In QA we can use this to get answers from long input texts by first getting predictions for smaller passages
        # and then aggregating them here.
        if aggregate_preds:
            # can assume that we have only complete docs i.e. all the samples of one doc are in the current chunk
            # TODO is there a better way than having to wrap logits all in list?
            # TODO can QA formatted preds deal with samples?
            preds_all = self.model.formatted_preds(logits=[logits_all],
                                                   baskets=baskets,
                                                   rest_api_schema=rest_api_schema)[0]
        return preds_all

    def extract_vectors(
        self, dicts, extraction_strategy="cls_token", extraction_layer=-1
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
