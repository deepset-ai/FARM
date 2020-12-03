import abc
import inspect
import json
import logging
import os
import random
from abc import ABC
from inspect import signature
from pathlib import Path
from random import randint
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers.configuration_auto import AutoConfig

from numpy.random import random as random_float
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.input_features import (
    samples_to_features_ner,
    samples_to_features_bert_lm,
    sample_to_features_text,
    sample_to_features_qa,
)
from farm.data_handler.samples import (
    Sample,
    SampleBasket,
    create_samples_qa
)

from farm.data_handler.utils import (
    pad,
    expand_labels,
    read_tsv,
    read_tsv_sentence_pair,
    read_docs_from_txt,
    read_ner_file,
    read_squad_file,
    read_jsonl,
    read_dpr_json,
    is_json,
    get_sentence_pair,
    split_with_metadata,
    convert_qa_input_dict,
    get_sequence_pair,
    join_sentences
)

from farm.data_handler.input_features import get_roberta_seq_2_start, get_camembert_seq_2_start

from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata, truncate_sequences, insert_at_special_tokens_pos
from farm.utils import MLFlowLogger as MlLogger
from farm.utils import try_get

ID_NAMES = ["example_id", "external_id", "doc_id", "id"]


logger = logging.getLogger(__name__)


class Processor(ABC):
    """
    Is used to generate PyTorch Datasets from input data. An implementation of this abstract class should be created
    for each new data source.
    Implement the abstract methods: file_to_dicts(), _dict_to_samples(), _sample_to_features()
    to be compatible with your data format
    """

    subclasses = {}

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        train_filename,
        dev_filename,
        test_filename,
        dev_split,
        data_dir,
        tasks={},
        proxies=None
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :type data_dir: str
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :type tasks: dict
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        """

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None
        self.baskets = []

        self._log_params()

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(
        cls,
        processor_name,
        data_dir,
        tokenizer,
        max_seq_len,
        train_filename,
        dev_filename,
        test_filename,
        dev_split,
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :type processor_name: str
        :param data_dir: Directory where data files are located.
        :type data_dir: str
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced.
                          Only works if dev_filename is set to None
        :type dev_split: float
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: An instance of the specified processor.
        """

        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(
            f"Got more parameters than needed for loading {processor_name}: {unused_args}. "
            f"Those won't be used!"
        )
        processor = cls.subclasses[processor_name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            **kwargs,
        )

        return processor

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Infers the specific type of Processor from a config file (e.g. GNADProcessor) and loads an instance of it.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. GNADProcessor)
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        config["inference"] = True
        # init tokenizer
        if "lower_case" in config.keys():
            logger.warning("Loading tokenizer from deprecated FARM config. "
                           "If you used `custom_vocab` or `never_split_chars`, this won't work anymore.")
            tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"], do_lower_case=config["lower_case"])
        else:
            tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name,
                               metric=task["metric"],
                               label_list=task["label_list"],
                               label_column_name=task["label_column_name"],
                               text_column_name=task.get("text_column_name", None),
                               task_type=task["task_type"])

        if processor is None:
            raise Exception

        return processor

    @classmethod
    def convert_from_transformers(cls, tokenizer_name_or_path, task_type, max_seq_len, doc_stride,
                                  tokenizer_class=None, tokenizer_args=None, use_fast=None):
        config = AutoConfig.from_pretrained(tokenizer_name_or_path)
        tokenizer_args = tokenizer_args or {}
        tokenizer = Tokenizer.load(tokenizer_name_or_path,
                                   tokenizer_class=tokenizer_class,
                                   use_fast=use_fast,
                                   **tokenizer_args,
                                   )

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
            label_list = list(config.id2label.values())
            processor = NERProcessor(
                tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir="data", metric="seq_f1",
                label_list=label_list
            )
        else:
            raise ValueError(f"`task_type` {task_type} is not supported yet. "
                             f"Valid options for arg `task_type`: 'question_answering', "
                             f"'embeddings', 'text_classification', 'ner'")

        return processor

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        # self.__dict__ doesn't give parent class attributes
        for key, value in inspect.getmembers(self):
            if is_json(value) and key[0] != "_":
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    def add_task(self, name,  metric, label_list, label_column_name=None,
                 label_name=None, task_type=None, text_column_name=None):
        if type(label_list) is not list:
            raise ValueError(f"Argument `label_list` must be of type list. Got: f{type(label_list)}")

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type
        }

    @abc.abstractmethod
    def file_to_dicts(self, file: str) -> [dict]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _dict_to_samples(cls, dictionary: dict, all_dicts=None) -> [Sample]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _sample_to_features(cls, sample: Sample) -> dict:
        raise NotImplementedError()

    def _dict_to_samples_and_features(self, dictionary: dict, all_dicts=None) -> [Sample]:
        raise NotImplementedError()

    def _init_samples_in_baskets(self):
        all_dicts = [b.raw for b in self.baskets]
        for basket in self.baskets:
            try:
                basket.samples = self._dict_to_samples(dictionary=basket.raw, all_dicts=all_dicts)
                for num, sample in enumerate(basket.samples):
                     sample.id = f"{basket.id_internal}-{num}"
            except Exception as e:
                logger.error(f"Could not create sample(s) from this dict: \n {basket.raw}")
                logger.error(f"Error message: {e}")

    def _featurize_samples(self):
        for basket in self.baskets:
            for sample in basket.samples:
                try:
                    sample.features = self._sample_to_features(sample=sample)
                except Exception as e:
                    logger.error(f"Could not convert this sample to features: \n {sample}")
                    logger.error(f"Basket id: id_internal: {basket.id_internal}, id_external: {basket.id_external}")
                    logger.error(f"Error message: {e}")

    def _init_and_featurize_samples_in_baskets(self):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
            try:
                basket.samples = self._dict_to_samples_and_features(dictionary=basket.raw,
                                                                    all_dicts=all_dicts,
                                                                    basket_id_internal=basket.id_internal)
                for num, sample in enumerate(basket.samples):
                    sample.id = f"{basket.id_internal}-{num}"
            except Exception as e:
                logger.error(f"Could not create sample(s) from this dict: \n {basket.raw}")
                logger.error(f"Error message: {e}")


    @staticmethod
    def _check_sample_features(basket):
        """Check if all samples in the basket has computed its features.

        Args:
            basket: the basket containing the samples

        Returns:
            True if all the samples in the basket has computed its features, False otherwise

        """
        for sample in basket.samples:
            if sample.features is None:
                return False
        return True

    def _create_dataset(self, keep_baskets=False):
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            # if basket_to_remove is not empty remove the related baskets
            logger.warning(f"Removing the following baskets because of errors in computing features:")
            for basket in basket_to_remove:
                logger.warning(f"Basket id: id_internal: {basket.id_internal}, id_external: {basket.id_external}")
                self.baskets.remove(basket)

        if not keep_baskets:
            # free up some RAM, we don't need baskets from here on
            self.baskets = None
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names

    def dataset_from_dicts(self, dicts, indices=None, return_baskets = False):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This can be used for inference mode.

        :param dicts: List of dictionaries where each contains the data of one input sample.
        :type dicts: list of dicts
        :return: a Pytorch dataset and a list of tensor names.
        """
        # We need to add the index (coming from multiprocessing chunks) to have a unique basket ID

        self.baskets = []
        for id_internal, d in enumerate(dicts):
            id_external = self._id_from_dict(d)
            if indices:
                id_internal = indices[id_internal]
            self.baskets.append(SampleBasket(raw=d, id_external=id_external, id_internal=id_internal))
        if self.tokenizer.is_fast:
            self._init_and_featurize_samples_in_baskets()
        else:
            self._init_samples_in_baskets()
            self._featurize_samples()
        if indices:
            if 0 in indices:
                self._log_samples(2)
        else:
            self._log_samples(2)
        if return_baskets:
            dataset, tensor_names = self._create_dataset(keep_baskets=True)
            return dataset, tensor_names, self.baskets
        else:
            dataset, tensor_names = self._create_dataset()
            return dataset, tensor_names

    def _log_samples(self, n_samples):
        logger.info("*** Show {} random examples ***".format(n_samples))
        for i in range(n_samples):
            random_basket = random.choice(self.baskets)
            random_sample = random.choice(random_basket.samples)
            logger.info(random_sample)


    def _log_params(self):
        params = {
            "processor": self.__class__.__name__,
            "tokenizer": self.tokenizer.__class__.__name__,
        }
        names = ["max_seq_len", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        MlLogger.log_params(params)

    @staticmethod
    def _id_from_dict(d):
        ext_id = try_get(ID_NAMES, d)
        if not ext_id and "qas" in d:
            ext_id = try_get(ID_NAMES, d["qas"][0])
        return ext_id


#########################################
# Processors for Text Classification ####
#########################################
class TextClassificationProcessor(Processor):
    """
    Used to handle the text classification datasets that come in tabular format (CSV, TSV, etc.)
    """
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param multilabel: set to True for multilabel classification
        :type multilabel: bool
        :param header: which line to use as a header in the input csv/tsv
        :type  header: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        #TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,

        )
        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(name="text_classification",
                          metric=metric,
                          label_list=label_list,
                          label_column_name=label_column_name,
                          text_column_name=text_column_name,
                          task_type=task_type)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {}
        for task in self.tasks.values():
            column_mapping[task["label_column_name"]] = task["label_name"]
            column_mapping[task["text_column_name"]] = "text"
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            rename_columns=column_mapping,
            header=self.header,
            proxies=self.proxies,
            max_samples=self.max_samples
            )

        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets and a start_of_word mask
        text = dictionary["text"]
        tokenized = tokenize_with_metadata(text, self.tokenizer)
        if len(tokenized["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None, tokenizer=self.tokenizer,
                                                max_seq_len=self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        text = dictionary["text"]
        inputs = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        # Get tokens as text with metadata
        tokens = []
        offsets = []
        start_of_word = []
        previous_token_end = -1
        for token_id, is_special_token, offset in zip(inputs["input_ids"],
                                                      inputs["special_tokens_mask"],
                                                      inputs["offset_mapping"]):
            if not is_special_token:
                tokens.append(self.tokenizer.convert_ids_to_tokens(token_id))
                offsets.append(offset[0])
                start_of_word.append(True if offset[0] != previous_token_end else False)
                previous_token_end = offset[1]

        token_dict = {"tokens": tokens,
                      "offsets": offsets,
                      "start_of_word": start_of_word}

        if len(token_dict["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []

        # Build feature dict
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)
        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)

        assert len(input_ids) == self.max_seq_len
        assert len(padding_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        feat_dict = {"input_ids": input_ids,
                     "padding_mask": padding_mask,
                     "segment_ids": segment_ids}

        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            try:
                label_name = task["label_name"]
                label_raw = dictionary[label_name]
                label_list = task["label_list"]
                if task["task_type"] == "classification":
                    # id of label
                    try:
                        label_ids = [label_list.index(label_raw)]
                    except ValueError as e:
                        raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
                elif task["task_type"] == "multilabel_classification":
                    # multi-hot-format
                    label_ids = [0] * len(label_list)
                    for l in label_raw.split(","):
                        if l != "":
                            label_ids[label_list.index(l)] = 1
                else:
                    raise ValueError(task["task_type"])
            except KeyError:
                # For inference mode we don't expect labels
                label_ids = None
            if label_ids is not None:
                feat_dict[task["label_tensor_name"]] = label_ids

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feat_dict])]

class TextPairClassificationProcessor(TextClassificationProcessor):
    """
    Used to handle text pair classification datasets (e.g. Answer Selection or Natural Inference) that come in
    tsv format. The columns should be called text, text_b and label.
    """
    def __init__(self, **kwargs):
        super(TextPairClassificationProcessor, self).__init__(**kwargs)

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
        dicts = read_tsv_sentence_pair(
            rename_columns=column_mapping,
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            proxies=self.proxies,
        )
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        tokenized_a = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        tokenized_b = tokenize_with_metadata(dictionary["text_b"], self.tokenizer)

        if len(tokenized_a["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []
        if len(tokenized_b["tokens"]) == 0:
            text_b = dictionary["text_b"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text_b}")
            return []

        tokenized = {"tokens": tokenized_a["tokens"],
                     "tokens_b": tokenized_b["tokens"]}
        tokenized["tokens"], tokenized["tokens_b"], _ = truncate_sequences(seq_a=tokenized["tokens"],
                                                                           seq_b=tokenized["tokens_b"],
                                                                           tokenizer=self.tokenizer,
                                                                           max_seq_len=self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        seq_a = dictionary["text"]
        seq_b = dictionary["text_b"]

        inputs = self.tokenizer.encode_plus(
            text=seq_a,
            text_pair=seq_b,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

        # Find position of [SEP]-token
        # seq_2_start_t is the index of the first token in the second text sequence (e.g. passage)
        if "RobertaTokenizer" in self.tokenizer.__class__.__name__:
            seq_2_start_t = get_roberta_seq_2_start(input_ids)
        elif "CamembertTokenizer" in self.tokenizer.__class__.__name__:
            seq_2_start_t = get_camembert_seq_2_start(input_ids)
        else:
            seq_2_start_t = segment_ids.index(1)

        # Get tokens as text with metadata
        tokens_a = []
        tokens_b = []
        for idx, (token_id, is_special_token) in enumerate(zip(input_ids,
                                                               inputs["special_tokens_mask"])):
            if not is_special_token:
                if idx < seq_2_start_t:
                    tokens_a.append(self.tokenizer.convert_ids_to_tokens(token_id))
                else:
                    tokens_b.append(self.tokenizer.convert_ids_to_tokens(token_id))

        token_dict = {"tokens": tokens_a,
                      "tokens_b": tokens_b}

        if len(token_dict["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {seq_a}")
            return []
        if len(token_dict["tokens_b"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {seq_b}")
            return []

        # Build feature dict

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)
        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)

        assert len(input_ids) == self.max_seq_len
        assert len(padding_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        feat_dict = {"input_ids": input_ids,
                     "padding_mask": padding_mask,
                     "segment_ids": segment_ids}

        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            try:
                label_name = task["label_name"]
                label_raw = dictionary[label_name]
                label_list = task["label_list"]
                if task["task_type"] == "classification":
                    # id of label
                    try:
                        label_ids = [label_list.index(label_raw)]
                    except ValueError as e:
                        raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
                elif task["task_type"] == "multilabel_classification":
                    # multi-hot-format
                    label_ids = [0] * len(label_list)
                    for l in label_raw.split(","):
                        if l != "":
                            label_ids[label_list.index(l)] = 1
                elif task["task_type"] == "regression":
                    label_ids = [float(label_raw)]
                else:
                    raise ValueError(task["task_type"])
            except KeyError:
                # For inference mode we don't expect labels
                label_ids = None
            if label_ids is not None:
                feat_dict[task["label_tensor_name"]] = label_ids

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feat_dict])]




#########################################
# Processors for Basic Inference ####
#########################################
class InferenceProcessor(Processor):
    """
    Generic processor used at inference time:
    - fast
    - no labels
    - pure encoding of text into pytorch dataset
    - Doesn't read from file, but only consumes dictionaries (e.g. coming from API requests)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        **kwargs,
    ):

        super(InferenceProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=None,
            dev_filename=None,
            test_filename=None,
            dev_split=None,
            data_dir=None,
            tasks={},
        )

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Overwriting method from parent class to **always** load the InferenceProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an InferenceProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"])
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name="InferenceProcessor", **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def file_to_dicts(self, file: str) -> [dict]:
        raise NotImplementedError

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None, tokenizer=self.tokenizer,
                                                max_seq_len=self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        text = dictionary["text"]
        inputs = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        # Get tokens as text with metadata
        tokens = []
        offsets = []
        start_of_word = []
        previous_token_end = -1
        for token_id, is_special_token, offset in zip(inputs["input_ids"],
                                                      inputs["special_tokens_mask"],
                                                      inputs["offset_mapping"]):
            if not is_special_token:
                tokens.append(self.tokenizer.convert_ids_to_tokens(token_id))
                offsets.append(offset[0])
                start_of_word.append(True if offset[0] != previous_token_end else False)
                previous_token_end = offset[1]

        token_dict = {"tokens": tokens,
                      "offsets": offsets,
                      "start_of_word": start_of_word}

        if len(token_dict["tokens"]) == 0:
            logger.warning(
                f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []

        # Build feature dict
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)
        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)

        assert len(input_ids) == self.max_seq_len
        assert len(padding_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        feat_dict = {"input_ids": input_ids,
                     "padding_mask": padding_mask,
                     "segment_ids": segment_ids}

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feat_dict])]


#########################################
# Processors for NER data ####
#########################################
class NERProcessor(Processor):
    """
    Used to handle most NER datasets, like CoNLL or GermEval 2014
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.txt",
        dev_filename="dev.txt",
        test_filename="test.txt",
        dev_split=0.0,
        delimiter="\t",
        proxies=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "seq_f1".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["seq_f1", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file. German version of Conll03 uses a whitespace. GermEval 2014 is tab separated \t
        :type delimiter: str
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        # Custom processor attributes
        self.delimiter = delimiter

        super(NERProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )

        if metric and label_list:
            self.add_task("ner", metric, label_list)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        dicts = read_ner_file(filename=file, sep=self.delimiter,  proxies=self.proxies)
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets, which helps to map our entity tags back to original positions
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        if len(tokenized["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None, tokenizer=self.tokenizer,
                                                max_seq_len=self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_ner(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        text = dictionary["text"]
        inputs = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        # Get tokens as text with metadata
        tokens = []
        offsets = []
        start_of_word = []
        previous_token_end = -1
        for token_id, is_special_token, offset in zip(inputs["input_ids"],
                                                      inputs["special_tokens_mask"],
                                                      inputs["offset_mapping"]):
            if not is_special_token:
                tokens.append(self.tokenizer.convert_ids_to_tokens(token_id))
                offsets.append(offset[0])
                start_of_word.append(True if offset[0] != previous_token_end else False)
                previous_token_end = offset[1]

        token_dict = {"tokens": tokens,
                      "offsets": offsets,
                      "start_of_word": start_of_word}

        if len(token_dict["tokens"]) == 0:
            logger.warning(
                f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []

        # Build feature dict
        input_ids = inputs["input_ids"]
        segment_ids = inputs["token_type_ids"]
        special_tokens_mask = inputs["special_tokens_mask"]

        # We construct a mask to identify the first token of a word. We will later only use them for predicting entities.
        # Special tokens don't count as initial tokens => we add 0 at the positions of special tokens
        # For BERT we add a 0 in the start and end (for CLS and SEP)
        initial_mask = [int(x) for x in token_dict["start_of_word"]]
        initial_mask = insert_at_special_tokens_pos(initial_mask, special_tokens_mask, insert_element=0)
        assert len(initial_mask) == len(input_ids)

        for task_name, task in self.tasks.items():
            try:
                label_list = task["label_list"]
                label_name = task["label_name"]
                label_tensor_name = task["label_tensor_name"]
                labels_word = dictionary[label_name]
                labels_token = expand_labels(labels_word, initial_mask, "X")
                label_ids = [label_list.index(lt) for lt in labels_token]
            except ValueError:
                label_ids = None
                problematic_labels = set(labels_token).difference(set(label_list))
                logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                               f"\nWe found a problem with labels {str(problematic_labels)}")
            except KeyError:
                # For inference mode we don't expect labels
                label_ids = None
                logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                               "\nIf your are running in *inference* mode: Don't worry!"
                               "\nIf you are running in *training* mode: Verify you are supplying a proper label list to "
                               "your processor and check that labels in input data are correct.")

        # This mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)

        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4" for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)
        initial_mask = pad(initial_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)
        if label_ids:
            label_ids = pad(label_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        feature_dict = {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
            "initial_mask": initial_mask,
        }

        if label_ids:
            feature_dict[label_tensor_name] = label_ids

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feature_dict])]


#####################
# LM Processors ####
#####################
class BertStyleLMProcessor(Processor):
    """
    Prepares data for masked language model training and next sentence prediction in the style of BERT
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename="train.txt",
        dev_filename="dev.txt",
        test_filename="test.txt",
        dev_split=0.0,
        next_sent_pred=True,
        next_sent_pred_style="sentence",
        max_docs=None,
        proxies=None,
        masked_lm_prob=0.15,
        
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param next_sent_pred: Whether to use next_sentence_prediction objective or not
        :type next_sent_pred: bool
        :param next_sent_pred_style:
            Two different styles for next sentence prediction available:
                - "sentence":   Use of a single sentence for Sequence A and a single sentence for Sequence B
                - "bert-style": Fill up all of max_seq_len tokens and split into Sequence A and B at sentence border.
                                If there are too many tokens, Sequence B will be truncated.
        :type next_sent_pred_style: str
        :param max_docs: maximum number of documents to include from input dataset
        :type max_docs: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param masked_lm_prob: probability of masking a token
        :type masked_lm_prob: float
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        self.delimiter = ""
        self.max_docs = max_docs

        super(BertStyleLMProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )

        self.next_sent_pred = next_sent_pred
        self.next_sent_pred_style = next_sent_pred_style
        added_tokens = self.get_added_tokens()
        self.add_task("lm", "acc", list(self.tokenizer.vocab) + added_tokens)
        if self.next_sent_pred:
            self.add_task("nextsentence", "acc", ["False", "True"])
        self.masked_lm_prob = masked_lm_prob


    def get_added_tokens(self):
        dictionary = self.tokenizer.get_added_vocab()
        sorted_tuples = sorted(dictionary.items(), key=lambda x: x[0])
        return [x[1] for x in sorted_tuples]

    def file_to_dicts(self, file: str) -> list:
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter, max_docs=self.max_docs, proxies=self.proxies)
        return dicts

    def _dict_to_samples(self, dictionary, all_dicts=None):
        doc = dictionary["doc"]

        # next sentence prediction...
        if self.next_sent_pred:
            assert len(all_dicts) > 1, "Need at least 2 documents to sample random sentences from"
            # ...with single sentences
            if self.next_sent_pred_style == "sentence":
                samples = self._dict_to_samples_single_sentence(doc, all_dicts)
            # ...bert style
            elif self.next_sent_pred_style == "bert-style":
                samples = self._dict_to_samples_bert_style(doc, all_dicts)
            else:
                raise NotImplementedError("next_sent_pred_style has to be 'sentence' or 'bert-style'")

        # no next sentence prediction
        else:
            samples = self._dict_to_samples_no_next_sent(doc)

        return samples

    def _dict_to_samples_single_sentence(self, doc, all_dicts):
        samples = []

        # create one sample for each sentence in the doc (except for the very last -> "nextSentence" is impossible)
        for idx in range(len(doc) - 1):
            tokenized = {}
            text_a, text_b, is_next_label = get_sentence_pair(doc, all_dicts, idx)
            sample_in_clear_text = {
                "text_a" : text_a,
                "text_b" : text_b,
                "nextsentence_label" : is_next_label,
            }
            # tokenize
            tokenized["text_a"] = tokenize_with_metadata(text_a, self.tokenizer)
            tokenized["text_b"] = tokenize_with_metadata(text_b, self.tokenizer)

            if len(tokenized["text_a"]["tokens"]) == 0:
                logger.warning(
                    f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text_a}")
                continue
            if len(tokenized["text_b"]["tokens"]) == 0:
                logger.warning(
                    f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text_b}")
                continue

            # truncate to max_seq_len
            for seq_name in ["tokens", "offsets", "start_of_word"]:
                tokenized["text_a"][seq_name], tokenized["text_b"][seq_name], _ = truncate_sequences(
                    seq_a=tokenized["text_a"][seq_name],
                    seq_b=tokenized["text_b"][seq_name],
                    tokenizer=self.tokenizer,
                    max_seq_len=self.max_seq_len)

            samples.append(Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized))

        return samples

    def _dict_to_samples_bert_style(self, doc, all_dicts):
        samples = []
        # account for [CLS], [SEP], [SEP]
        max_num_tokens = self.max_seq_len - 3

        # tokenize
        doc_tokenized = []
        for sentence in doc:
            doc_tokenized.append(tokenize_with_metadata(sentence, self.tokenizer))

        current_chunk = []
        current_chunk_clear_text = []
        current_length = 0
        i = 0
        while i < len(doc_tokenized):
            current_segment = doc_tokenized[i]
            current_length += len(current_segment["tokens"])
            current_chunk.append(current_segment)
            current_chunk_clear_text.append(doc[i])

            # reached end of document or max_num_tokens
            if (i == len(doc_tokenized) - 1) or (current_length >= max_num_tokens):
                sequence_a, sequence_b, sample_in_clear_text, num_unused_segments = get_sequence_pair(
                    doc,
                    current_chunk,
                    current_chunk_clear_text,
                    all_dicts,
                    self.tokenizer,
                    max_num_tokens,
                )

                sequence_a = join_sentences(sequence_a)
                sequence_b = join_sentences(sequence_b)
                for seq_name in ["tokens", "offsets", "start_of_word"]:
                    sequence_a[seq_name], sequence_b[seq_name], _ = truncate_sequences(
                        seq_a=sequence_a[seq_name],
                        seq_b=sequence_b[seq_name],
                        tokenizer=self.tokenizer,
                        max_seq_len=max_num_tokens,
                        with_special_tokens=False,
                        truncation_strategy="only_second",
                    )
                tokenized = {"text_a" : sequence_a, "text_b" : sequence_b}
                samples.append(Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized))

                i -= num_unused_segments

                current_chunk = []
                current_chunk_clear_text = []
                current_length = 0
            i += 1
        return samples

    def _dict_to_samples_no_next_sent(self, doc):
        samples = []

        for idx in range(len(doc)):
            tokenized = {}
            text_a = doc[idx]
            sample_in_clear_text = {
                "text_a": text_a,
                "text_b": None,
                "nextsentence_label": None,
            }
            # tokenize
            tokenized["text_a"] = tokenize_with_metadata(
                text_a, self.tokenizer
            )
            if len(tokenized["text_a"]["tokens"]) == 0:
                continue
            # truncate to max_seq_len
            for seq_name in ["tokens", "offsets", "start_of_word"]:
                tokenized["text_a"][seq_name], _, _ = truncate_sequences(
                    seq_a=tokenized["text_a"][seq_name],
                    seq_b=None,
                    tokenizer=self.tokenizer,
                    max_seq_len=self.max_seq_len,
                )

            samples.append(Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized))

        return samples

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_bert_lm(
            sample=sample, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer,
            next_sent_pred=self.next_sent_pred, masked_lm_prob=self.masked_lm_prob
        )
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, all_dicts=None, **kwargs) -> [Sample]:
        doc = dictionary["doc"]

        # Initialize samples
        # next sentence prediction...
        if self.next_sent_pred:
            assert len(all_dicts) > 1, "Need at least 2 documents to sample random sentences from"
            # ...with single sentences
            if self.next_sent_pred_style == "sentence":
                samples = self._dict_to_samples_single_sentence(doc, all_dicts)
            # ...bert style
            elif self.next_sent_pred_style == "bert-style":
                samples = self._dict_to_samples_bert_style(doc, all_dicts)
            else:
                raise NotImplementedError("next_sent_pred_style has to be 'sentence' or 'bert-style'")

        # no next sentence prediction
        else:
            samples = self._dict_to_samples_no_next_sent(doc)

        # Get features
        for sample in samples:
            features = self._sample_to_features(sample)
            sample.features = features

        return samples

    def estimate_n_samples(self, filepath, max_docs=500):
        """
        Estimates the number of samples from a given file BEFORE preprocessing.
        Used in StreamingDataSilo to estimate the number of steps before actually processing the data.
        The estimated number of steps will impact some types of Learning Rate Schedules.
        :param filepath: str or Path, file with data used to create samples (e.g. train.txt)
        :param max_docs: int, maximum number of docs to read in & use for our estimate of n_samples
        :return: int, number of samples in the given dataset
        """

        total_lines = sum(1 for line in open(filepath, encoding="utf-8"))
        empty_lines = sum(1 if line == "\n" else 0 for line in open(filepath, encoding="utf-8"))

        if self.next_sent_pred_style == "sentence":
            # one sample = two lines (except last line in doc)
            n_samples = total_lines - (2 * empty_lines)
        elif self.next_sent_pred_style == "bert-style":
            # Original BERT LM training (filling up sequence pairs with sentences until max_seq_len)
            # (This is a very rough heuristic, as we can only estimate the real number of samples AFTER tokenization)
            logging.info(f"Estimating total number of samples ...")
            # read in subset of docs
            if self.max_docs:
                temp = self.max_docs
                self.max_docs = min(max_docs, temp)
                dicts = list(self.file_to_dicts(filepath))
                self.max_docs = temp
            else:
                self.max_docs = max_docs
                dicts = list(self.file_to_dicts(filepath))
                self.max_docs = None
            # count samples
            n_samples = 0
            for d in dicts:
                n_samples += len(self._dict_to_samples_bert_style(doc=d["doc"], all_dicts=dicts))
            n_samples = int(n_samples / len(dicts)) * (empty_lines+1)
            logging.info(f"Heuristic estimate of number of samples in {filepath} based on {len(dicts)} docs: {n_samples}")
        else:
            raise NotImplementedError(f"No estimate logic for next_sent_pred_style={self.next_sent_pred_style} implemented")
        return n_samples


#########################################
# QA Processors ####
#########################################

class QAProcessor(Processor):
    """
    This is class inherits from Processor and is the parent to SquadProcessor and NaturalQuestionsProcessor.
    Its main role is to extend the __init__() so that the number of starting, intermediate and end special tokens
    are calculated from the tokenizer and store as attributes. These are used by the child processors in their
    sample_to_features() methods
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_special_tokens_count()

    def initialize_special_tokens_count(self):
        vec = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=["a"],
                                                              token_ids_1=["b"])
        self.sp_toks_start = vec.index("a")
        self.sp_toks_mid = vec.index("b") - self.sp_toks_start - 1
        self.sp_toks_end = len(vec) - vec.index("b") - 1


class SquadProcessor(QAProcessor):
    """ Used to handle the SQuAD dataset"""

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric="squad",
        train_filename=Path("train-v2.0.json"),
        dev_filename=Path("dev-v2.0.json"),
        test_filename=None,
        dev_split=0,
        doc_stride=128,
        max_query_length=64,
        proxies=None,
        max_answers=6,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, can be "squad" or "top_n_accuracy"
        :type metric: str
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param doc_stride: When the document containing the answer is too long it gets split into part, strided by doc_stride
        :type doc_stride: int
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :type max_query_length: int
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        self.target = "classification"
        self.ph_output_type = "per_token_squad"

        assert doc_stride < max_seq_len, "doc_stride is longer than max_seq_len. This means that there will be gaps " \
                                         "as the passage windows slide, causing the model to skip over parts of the document. "\
                                         "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384) "

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_answers = max_answers

        super(SquadProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )
        if metric and label_list:
            self.add_task("question_answering", metric, label_list)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        """ Overwrites the method from the base class since Question Answering processing is quite different.
        This method allows for documents and questions to be tokenized earlier. Then SampleBaskets are initialized
        with one document and one question. """

        dicts = [convert_qa_input_dict(x) for x in dicts]
        if self.tokenizer.is_fast:
            self.baskets = self._dicts_to_baskets_samples_and_features(dicts, indices)
        else:
            self.baskets = self._dicts_to_baskets(dicts, indices)
            self._init_samples_in_baskets()
            self._featurize_samples()
        if 0 in indices:
            self._log_samples(2)
        # This mode is for inference where we need to keep baskets
        if return_baskets:
            dataset, tensor_names = self._create_dataset(keep_baskets=True)
            return dataset, tensor_names, self.baskets
        # This mode is for training where we can free ram by removing baskets
        else:
            dataset, tensor_names = self._create_dataset(keep_baskets=False)
            return dataset, tensor_names

    def _dicts_to_baskets(self, dicts, indices):
        # Perform tokenization on documents and questions resulting in an unnested list of doc-question pairs
        dicts_tokenized = [_apply_tokenization(d, self.tokenizer) for d in dicts]

        baskets = []

        for index, document in zip(indices, dicts_tokenized):
            for q_idx, raw in enumerate(document):
                # TODO: These checks dont exist in NQProcessor
                # ignore samples with empty context
                if raw["document_text"] == "":
                    logger.warning("Ignoring sample with empty context.")
                    continue

                # Removes answers where text = "". True no_answers should have raw["answers"] = []
                raw["answers"] = [a for a in raw["answers"] if a["text"]]

                # check if answer string can be found in context
                for answer in raw["answers"]:
                    if answer["text"] not in raw["document_text"]:
                        logger.warning(f"Answer '{answer['text']}' not contained in context.")
                # In case of Question Answering the external ID is used for document IDs
                id_external = try_get(ID_NAMES, raw)
                id_internal = f"{index}-{q_idx}"
                basket = SampleBasket(raw=raw, id_internal=id_internal, id_external=id_external)
                baskets.append(basket)
        return baskets

    def file_to_dicts(self, file: str) -> [dict]:
        nested_dicts = read_squad_file(filename=file)
        dicts = [y for x in nested_dicts for y in x["paragraphs"]]
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        samples = create_samples_qa(dictionary=dictionary,
                                       max_query_len=self.max_query_length,
                                       max_seq_len=self.max_seq_len,
                                       doc_stride=self.doc_stride,
                                       n_special_tokens=n_special_tokens)
        return samples

    def _sample_to_features(self, sample) -> dict:
        _check_valid_answer(sample)
        features = sample_to_features_qa(sample=sample,
                                         tokenizer=self.tokenizer,
                                         max_seq_len=self.max_seq_len,
                                         sp_toks_start=self.sp_toks_start,
                                         sp_toks_mid=self.sp_toks_mid,
                                         sp_toks_end=self.sp_toks_end,
                                         max_answers=self.max_answers)
        return features

    def _dicts_to_baskets_samples_and_features(self, dicts, indices) -> [SampleBasket]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        # Perform tokenization on documents and questions resulting in an unnested list of doc-question pairs
        dicts_tokenized = [_apply_tokenization(d, self.tokenizer) for d in dicts]
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)

        baskets = []

        for index, document in zip(indices, dicts_tokenized):
            for q_idx, raw in enumerate(document):
                # ignore samples with empty context
                if raw["document_text"] == "":
                    logger.warning("Ignoring sample with empty context")
                    continue
                # check if answer string can be found in context
                for answer in raw["answers"]:
                    if answer["text"] not in raw["document_text"]:
                        logger.warning(f"Answer '{answer['text']}' not contained in context.")
                # In case of Question Answering the external ID is used for document IDs
                id_external = try_get(ID_NAMES, raw)
                id_internal = f"{index}-{q_idx}"

                # create out of each Question-Document pair samples of Question-Passage pairs
                samples = create_samples_qa(dictionary=raw,
                                            max_query_len=self.max_query_length,
                                            max_seq_len=self.max_seq_len,
                                            doc_stride=self.doc_stride,
                                            n_special_tokens=n_special_tokens)
                # Add features to samples
                for num, sample in enumerate(samples):
                    sample.id = f"{id_internal}-{num}"
                    sample.features = self._sample_to_features(sample)
                basket = SampleBasket(raw=raw, id_internal=id_internal, id_external=id_external, samples=samples)
                baskets.append(basket)

        return baskets


class NaturalQuestionsProcessor(QAProcessor):
    """ Used to handle the Natural Question QA dataset"""

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename=Path("train-v2.0.json"),
        dev_filename=Path("dev-v2.0.json"),
        test_filename=None,
        dev_split=0,
        doc_stride=128,
        max_query_length=64,
        proxies=None,
        keep_no_answer=0.02,
        downsample_context_size=None,
        inference=False,
        max_answers=6,
        **kwargs):
        """
        Deals with all the preprocessing steps needed for Natural Questions. Follows Alberti 2019 et al. (https://arxiv.org/abs/1901.08634)
        in merging multiple disjoint short answers into the one longer label span and also by downsampling
        samples of no_answer during training

        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing the test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param doc_stride: When the document containing the answer is too long it gets split into parts, strided by doc_stride
        :type doc_stride: int
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :type max_query_length: int
        :param keep_no_answer: The probability that a sample with an no_answer label is kept
                                    (0.0 < keep_no_answer <= 1.0). Only works if inference is False
        :type keep_no_answer: float
        :param downsample_context_size: Downsampling before any data conversion by taking a short text window of size
                                        downsample_context_size around the long answer span. To disable set to None
        :type downsample_context_size: int
        :param inference: Whether we are currently using the Processsor for model inference. If True, the
                          keep_no_answer will be overridden and set to 1
        :type inference: bool
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        self.target = "classification"
        self.ph_output_type = "per_token_squad"

        # These are classification labels from Natural Questions. Note that in this implementation, we are merging
        # the "long_answer" and "short_answer" labels into the one "span" label
        self.answer_type_list = ["no_answer", "span", "yes", "no"]

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.keep_no_answer = keep_no_answer
        self.downsample_context_size = downsample_context_size
        self.inference = inference
        self.max_answers = max_answers

        super(NaturalQuestionsProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )

        # Todo rename metric from squad to maybe QA spans or something like that
        self.add_task("question_answering", "squad", ["start_token", "end_token"])
        self.add_task("text_classification", "f1_macro", self.answer_type_list, label_name="answer_type")

    def file_to_dicts(self, file: str) -> [dict]:
        dicts = read_jsonl(file, proxies=self.proxies)
        return dicts

    def _dict_to_samples(self, dictionary: dict, all_dicts=None) -> [Sample]:
        """
            This method will split question-document pairs from the SampleBasket into question-passage pairs which will
        each form one sample. The "t" and "c" in variables stand for token and character respectively. This uses many
        methods that the SquadProcessor calls but note that the SquadProcessor overwrites Processor._dicts_to_baskets()
        while the NaturalQuestionsProcessor does not. This was done in Squad to avoid retokenizing documents that are
        paired with multiple questions. This is not necessary for Natural Questions since there is generally a 1 to 1
        mapping from document to question. Input dictionaries can have either ["context", "qas"] (internal format) as
        keys or ["text", "questions"] (api format). Both are supported.
        """
        # Turns NQ dictionaries into a SQuAD style dictionaries
        if self._is_nq_dict(dictionary):
            dictionary = self._prepare_dict(dictionary=dictionary)

        dictionary_tokenized = _apply_tokenization(dictionary, self.tokenizer, self.answer_type_list)[0]
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        samples = create_samples_qa(dictionary_tokenized,
                                    self.max_query_length,
                                    self.max_seq_len,
                                    self.doc_stride,
                                    n_special_tokens)
        # Downsample the number of samples with an no_answer label. This fn will always return at least one sample
        # so that we don't end up with a basket with 0 samples
        if not self.inference:
            samples = self._downsample(samples, self.keep_no_answer)
        return samples

    @staticmethod
    def _is_nq_dict(dictionary):
        if set(dictionary.keys()) == {'document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'}:
            return True
        return False

    def _downsample(self, samples, keep_prob):
        # Downsamples samples with a no_answer label (since there is an overrepresentation of these in NQ)
        # This method will always return at least one sample. This is done so that we don't end up with SampleBaskets
        # with 0 samples
        ret = []
        for s in samples:
            if self._check_no_answer_sample(s):
                if random_float() > 1 - keep_prob:
                    ret.append(s)
            else:
                ret.append(s)
        if len(ret) == 0:
            ret = [random.choice(samples)]
        return ret

    def _downsample_unprocessed(self, dictionary):
        doc_text = dictionary["document_text"]
        doc_tokens = doc_text.split(" ")
        annotations = dictionary.get("annotations",[])
        # for simplicity we only downsample wiki pages with one long answer annotation
        if len(annotations) == 1:
            annotation = annotations[0]
            # There seem to be cases where there is no answer but an annotation is given as a (-1, -1) long answer
            if self._check_no_answer(annotation):
                dictionary["document_text"] = " ".join(doc_tokens[:self.max_seq_len+randint(1,self.downsample_context_size)])
            else:
                # finding earliest start and latest end labels
                long_answer_start = annotation['long_answer']['start_token']
                long_answer_end = annotation['long_answer']['end_token']
                short_answer_start = 1e10
                short_answer_end = -1
                for s in annotation["short_answers"]:
                    if s["start_token"] < short_answer_start:
                        short_answer_start = s["start_token"]
                    if s["end_token"] > short_answer_end:
                        short_answer_end = s["end_token"]

                start_threshold = min(long_answer_start,short_answer_start) - randint(1,self.downsample_context_size)
                start_threshold = max(0, start_threshold)
                end_threshold = max(long_answer_end,short_answer_end) + randint(1,self.downsample_context_size)

                # taking subset of doc text and shift labels
                sub_document_text = " ".join(
                    doc_tokens[start_threshold:end_threshold]
                )
                dictionary["document_text"] = sub_document_text
                # change of offsets happens in place (of dictionary)
                annotation['long_answer']['start_token'] -= start_threshold
                annotation['long_answer']['end_token'] -= start_threshold
                for s in annotation["short_answers"]:
                    s["start_token"] -= start_threshold
                    s["end_token"] -= start_threshold

        return dictionary


    def _prepare_dict(self, dictionary):
        """ Casts a Natural Questions dictionary that is loaded from a jsonl file into SQuAD format so that
        the same featurization functions can be called for both tasks. Each annotation can be one of four answer types,
        ["yes", "no", "span", "no_answer"]"""

        if self.downsample_context_size is not None:
            dictionary = self._downsample_unprocessed(dictionary)

        converted_answers = []
        doc_text = dictionary["document_text"]
        _, tok_to_ch = split_with_metadata(doc_text)
        for annotation in dictionary["annotations"]:
            # There seem to be cases where there is no answer but an annotation is given as a (-1, -1) long answer
            if self._check_no_answer(annotation):
                continue
            sa_text, sa_start_c = self._unify_short_answers(annotation["short_answers"], doc_text, tok_to_ch)
            la_text, la_start_c = self._retrieve_long_answer(annotation["long_answer"]["start_token"],
                                                             annotation["long_answer"]["end_token"],
                                                             tok_to_ch,
                                                             doc_text)
            # Picks the span to be considered as annotation by choosing between short answer, long answer and no_answer
            text, start_c = self._choose_span(sa_text, sa_start_c, la_text, la_start_c)
            converted_answers.append({"text": text,
                                      "answer_start": start_c})
        if len(converted_answers) == 0:
            answer_type = "no_answer"
        else:
            answer_type = dictionary["annotations"][0]["yes_no_answer"].lower()
            if answer_type == "none":
                answer_type = "span"
        # TODO: answer_type should be in answers since in NQ, each annotator can give either a span, no_answer, yes or no
        converted = {"id": dictionary["example_id"],
                     "context": doc_text,
                     "qas": [{"question": dictionary["question_text"],
                              "id": dictionary["example_id"],
                              "answers": converted_answers,
                              "answer_type": answer_type}]}
        return converted

    @staticmethod
    def _check_no_answer(annotation):
        if annotation["long_answer"]["start_token"] > -1 or annotation["long_answer"]["end_token"] > -1:
            return False
        for sa in annotation["short_answers"]:
            if sa["start_token"] > -1 or sa["end_token"] > -1:
                return False
        else:
            return True

    @staticmethod
    def _check_no_answer_sample(sample):
        sample_tok = sample.tokenized
        if len(sample_tok["answers"]) == 0:
            return True
        first_answer = sample_tok["answers"][0]
        if first_answer["start_t"] < sample_tok["passage_start_t"]:
            return True
        if first_answer["end_t"] > sample_tok["passage_start_t"] + len(sample_tok["passage_tokens"]):
            return True
        if first_answer["answer_type"] == "no_answer":
            return True
        else:
            return False

    def _retrieve_long_answer(self, start_t, end_t, tok_to_ch, doc_text):
        """ Retrieves the string long answer and also its starting character index"""
        start_c, end_c = self._convert_tok_to_ch(start_t, end_t, tok_to_ch, doc_text)
        text = doc_text[start_c: end_c]
        return text, start_c

    @staticmethod
    def _choose_span(sa_text, sa_start_c, la_text, la_start_c):
        if sa_text:
            return sa_text, sa_start_c
        elif la_text:
            return la_text, la_start_c
        else:
            return "", -1

    def _unify_short_answers(self, short_answers, doc_text, tok_to_ch):
        """ In cases where an NQ sample has multiple disjoint short answers, this fn generates the single shortest
        span that contains all the answers"""
        if not short_answers:
            return "", -1
        short_answer_idxs = []
        # TODO write comment explaining this
        for short_answer in short_answers:
            short_answer_idxs.append(short_answer["start_token"])
            short_answer_idxs.append(short_answer["end_token"])
        answer_start_t = min(short_answer_idxs)
        answer_end_t = max(short_answer_idxs)
        answer_start_c, answer_end_c = self._convert_tok_to_ch(answer_start_t, answer_end_t, tok_to_ch, doc_text)
        answer_text = doc_text[answer_start_c: answer_end_c]
        assert answer_text == " ".join(doc_text.split()[answer_start_t: answer_end_t])
        return answer_text, answer_start_c

    @staticmethod
    def _convert_tok_to_ch(start_t, end_t, tok_to_ch, doc_text):
        n_tokens = len(tok_to_ch)
        if start_t == -1 and end_t == -1:
            return -1, -1
        start_c = tok_to_ch[start_t]
        # when the end of the answer span is the end of the text
        if end_t == n_tokens:
            end_c = len(doc_text)
        else:
            next_word_start_c = tok_to_ch[end_t]
            span = doc_text[:next_word_start_c].strip()
            end_c = len(span)
        return start_c, end_c

    def _sample_to_features(self, sample: Sample) -> dict:
        _check_valid_answer(sample)
        features = sample_to_features_qa(sample=sample,
                                         tokenizer=self.tokenizer,
                                         max_seq_len=self.max_seq_len,
                                         sp_toks_start=self.sp_toks_start,
                                         sp_toks_mid=self.sp_toks_mid,
                                         sp_toks_end=self.sp_toks_end,
                                         answer_type_list=self.answer_type_list,
                                         max_answers=self.max_answers)
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """
            This method will split the question-document pair from the dictionary into question-passage pairs which will
        each form one sample. The "t" and "c" in variables stand for token and character respectively.
        Input dictionaries can have either ["context", "qas"] (internal format) as keys or ["text", "questions"]
        (api format). Both are supported.
        """
        if self._is_nq_dict(dictionary):
            dictionary = self._prepare_dict(dictionary=dictionary)
        basket_id_internal = kwargs["basket_id_internal"]

        dictionary_tokenized = _apply_tokenization(dictionary, self.tokenizer, self.answer_type_list)[0]
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        samples = create_samples_qa(dictionary_tokenized,
                                    self.max_query_length,
                                    self.max_seq_len,
                                    self.doc_stride,
                                    n_special_tokens)
        # Downsample the number of samples with an no_answer label. This fn will always return at least one sample
        # so that we don't end up with a basket with 0 samples.
        if not self.inference:
            samples = self._downsample(samples, self.keep_no_answer)

        # Get features for each sample
        for num, sample in enumerate(samples):
            sample.id = f"{basket_id_internal}-{num}"
            features = self._sample_to_features(sample)
            sample.features = features

        return samples


class RegressionProcessor(Processor):
    """
    Used to handle a regression dataset in tab separated text + label
    """
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        label_name="regression_label",
        scaler_mean=None,
        scaler_scale=None,
        proxies=None,
        text_column_name="text",
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a
                 numerical value. For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param label_name: name for the internal label variable in FARM (only needed to adjust in rare cases)
        :type label_name: str
        :param scaler_mean: Value to substract from the label for normalization
        :type scaler_mean: float
        :param scaler_scale: Value to divide the label by for normalization
        :type scaler_scale: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows

        super(RegressionProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            proxies=proxies
        )

        # Note that label_list is being hijacked to store the scaling mean and scale
        self.add_task(name="regression",
                      metric="mse",
                      label_list=[scaler_mean, scaler_scale],
                      label_column_name=label_column_name,
                      task_type="regression",
                      label_name=label_name,
                      text_column_name=text_column_name)

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {}
        for task in self.tasks.values():
            column_mapping[task["label_column_name"]] = task["label_name"]
            column_mapping[task["text_column_name"]] = "text"
        dicts = read_tsv(
            rename_columns=column_mapping,
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            proxies=self.proxies
        )

        # collect all labels and compute scaling stats
        train_labels = []
        if self.train_filename in str(file):
            for d in dicts:
                train_labels.append(float(d[self.tasks["regression"]["label_name"]]))
            scaler = StandardScaler()
            scaler.fit(np.reshape(train_labels, (-1, 1)))
            # add to label list in regression task
            self.tasks["regression"]["label_list"] = [scaler.mean_.item(), scaler.scale_.item()]

        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        if len(tokenized["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None,
                                                           tokenizer=self.tokenizer,
                                                           max_seq_len=self.max_seq_len)
        # Samples don't have labels during Inference mode
        label_key = self.tasks["regression"]["label_name"]
        if label_key in dictionary:
            label = float(dictionary[label_key])
            scaled_label = (label - self.tasks["regression"]["label_list"][0]) / self.tasks["regression"]["label_list"][1]
            dictionary[label_key] = scaled_label
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer
        )
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        text = dictionary["text"]
        inputs = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        # Get tokens as text with metadata
        tokens = []
        offsets = []
        start_of_word = []
        previous_token_end = -1
        for token_id, is_special_token, offset in zip(inputs["input_ids"],
                                                      inputs["special_tokens_mask"],
                                                      inputs["offset_mapping"]):
            if not is_special_token:
                tokens.append(self.tokenizer.convert_ids_to_tokens(token_id))
                offsets.append(offset[0])
                start_of_word.append(True if offset[0] != previous_token_end else False)
                previous_token_end = offset[1]

        token_dict = {"tokens": tokens,
                      "offsets": offsets,
                      "start_of_word": start_of_word}

        if len(token_dict["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []

        # Build feature dict
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)
        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)

        assert len(input_ids) == self.max_seq_len
        assert len(padding_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        feat_dict = {"input_ids": input_ids,
                     "padding_mask": padding_mask,
                     "segment_ids": segment_ids}

        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            try:
                label_name = task["label_name"]
                label_raw = dictionary[label_name]
                if task["task_type"] == "regression":
                    label_ids = [float(label_raw)]
                else:
                    raise ValueError(task["task_type"])
            except KeyError:
                # For inference mode we don't expect labels
                label_ids = None
            if label_ids is not None:
                feat_dict[task["label_tensor_name"]] = label_ids

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feat_dict])]


class TextSimilarityProcessor(Processor):
    """
    Used to handle the text DPR datasets that come in json format, example: nq-train.json, nq-dev.json, trivia-train.json, trivia-dev.json
    Datasets can be downloaded from the official DPR github repository (https://github.com/facebookresearch/DPR)

    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {"dataset": str,
    "question": str,
    "answers": list of str
    "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    Example of 1 sample in DPR data json:
    {
    "dataset": "nq_dev_psgs_w100",
    "question": "who sings does he love me with reba",
    "answers": ["Linda Davis"],
    "positive_ctxs": [
    {
    "title": "Does He Love You",
    "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
    "score": 1000,
    "title_score": 1,
    "passage_id": "11828866"
    },
    {
    "title": "Does He Love You",
    "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
    "score": 13.394315,
    "title_score": 0,
    "passage_id": "11828866"
    }, .... ]
    "negative_ctxs": [
    {
    "title": "Cormac McCarthy",
    "text": "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from \"The New York Times\", \"McCarthy doesn't drink anymore \u2013 he quit 16 years ago in El Paso, with one of his young",
    "score": 0,
    "title_score": 0,
    "passage_id": "2145653"
    },
    {
    "title": "Pragmatic Sanction of 1549",
    "text": "one heir, Charles effectively united the Netherlands as one entity. After Charles' abdication in 1555, the Seventeen Provinces passed to his son, Philip II of Spain. The Pragmatic Sanction is said to be one example of the Habsburg contest with particularism that contributed to the Dutch Revolt. Each of the provinces had its own laws, customs and political practices. The new policy, imposed from the outside, angered many inhabitants, who viewed their provinces as distinct entities. It and other monarchical acts, such as the creation of bishoprics and promulgation of laws against heresy, stoked resentments, which fired the eruption of",
    "score": 0,
    "title_score": 0,
    "passage_id": "2271902"
    }, ..... ]
    "hard_negative_ctxs": [
    {
    "title": "Why Don't You Love Me (Beyonce\u0301 song)",
    "text": "song. According to the lyrics of \"Why Don't You Love Me\", Knowles impersonates a woman who questions her love interest about the reason for which he does not value her fabulousness, convincing him she's the best thing for him as she sings: \"Why don't you love me... when I make me so damn easy to love?... I got beauty... I got class... I got style and I got ass...\". The singer further tells her love interest that the decision not to choose her is \"entirely foolish\". Originally released as a pre-order bonus track on the deluxe edition of \"I Am...",
    "score": 14.678405,
    "title_score": 0,
    "passage_id": "14525568"
    },
    {
    "title": "Does He Love You",
    "text": "singing the second chorus. Reba stays behind the wall the whole time, while Linda is in front of her. It then briefly goes back to the dressing room, where Reba continues to smash her lover's picture. The next scene shows Reba approaching Linda's house in the pouring rain at night, while Linda stands on her porch as they sing the bridge. The scene then shifts to the next day, where Reba watches from afar as Linda and the man are seen on a speedboat, where he hugs her, implying that Linda is who he truly loves. Reba finally smiles at",
    "score": 14.385411,
    "title_score": 0,
    "passage_id": "11828871"
    }, ...]
    """
    def __init__(
        self,
        tokenizer,
        passage_tokenizer,
        max_seq_len_query,
        max_seq_len_passage,
        data_dir="",
        metric=None,
        train_filename="train.json",
        dev_filename=None,
        test_filename="test.json",
        dev_split=0.1,
        proxies=None,
        max_samples=None,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
        shuffle_negatives=True,
        shuffle_positives=False,
        label_list=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :type max_seq_len_query: int
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :type max_seq_len_passage: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param max_samples: maximum number of samples to use
        :type max_samples: int
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :type shuffle_negatives: bool
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :type shuffle_positives: bool
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :type label_list: list[str]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        #TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.embed_title = embed_title
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage

        super(TextSimilarityProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len_query,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(name="text_similarity",
                          metric=metric,
                          label_list=label_list,
                          label_name="label",
                          task_type="text_similarity")
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Overwriting method from parent class to **always** load the TextSimilarityProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an TextSimilarityProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"])
        passage_tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["passage_tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]
        del config["passage_tokenizer"]

        processor = cls.load(tokenizer=tokenizer, passage_tokenizer=passage_tokenizer, processor_name="TextSimilarityProcessor", **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__
        config["passage_tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))
        self.passage_tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def file_to_dicts(self, file: str) -> [dict]:
        """
        Converts a Dense Passage Retrieval (DPR) data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format

        Returns:
        list of dictionaries: List[dict]
        each dictionary:
        {"query": str,
        "passages": [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
        {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
        ...]}
        """
        dicts = read_dpr_json(file, max_samples=self.max_samples)
        return dicts

    def _normalize_question(self, question: str) -> str:
        """
        Removes '?' from queries/questions

        :param question: string representing the question

        Returns:
            Question without the '?'
        """
        if question[-1] == '?':
            question = question[:-1]
        return question

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        Creates one sample from one dict consisting of the query, positive passages and hard negative passages
        :param dictionary:  {"query": str,
                            "passages": List[
                                            {'title': str,
                                            'text': str,
                                            'label': 'hard_negative',
                                            'external_id': str},
                                            {'title': str,
                                            'text': str,
                                            'label': 'positive',
                                            'external_id': str},
                                            ....
                                            ]
                            }

        Returns:
                sample: instance of Sample
        """


        clear_text = {}
        tokenized = {}
        features = {}
        # extract query, positive context passages and titles, hard-negative passages and titles
        if "query" in dictionary.keys():
            query = self._normalize_question(dictionary["query"])

            # featurize the query
            query_inputs = self.query_tokenizer.encode_plus(
                text=query,
                max_length=self.max_seq_len_query,
                add_special_tokens=True,
                truncation=True,
                truncation_strategy='longest_first',
                padding="max_length",
                return_token_type_ids=True,
            )

            query_input_ids, query_segment_ids, query_padding_mask = query_inputs["input_ids"], query_inputs[
                "token_type_ids"], query_inputs["attention_mask"]

            # tokenize query
            tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_input_ids)

            if len(tokenized_query) == 0:
                logger.warning(
                    f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize")
                return None

            clear_text["query_text"] = query
            tokenized["query_tokens"] = tokenized_query
            features["query_input_ids"] = query_input_ids
            features["query_segment_ids"] = query_segment_ids
            features["query_attention_mask"] = query_padding_mask

        if "passages" in dictionary.keys():
            positive_context = list(filter(lambda x: x["label"] == "positive", dictionary["passages"]))
            if self.shuffle_positives:
                random.shuffle(positive_context)
            positive_context = positive_context[:self.num_positives]
            hard_negative_context = list(filter(lambda x: x["label"] == "hard_negative", dictionary["passages"]))
            if self.shuffle_negatives:
                random.shuffle(hard_negative_context)
            hard_negative_context = hard_negative_context[:self.num_hard_negatives]

            positive_ctx_titles = [passage.get("title", None) for passage in positive_context]
            positive_ctx_texts = [passage["text"] for passage in positive_context]
            hard_negative_ctx_titles = [passage.get("title", None) for passage in hard_negative_context]
            hard_negative_ctx_texts = [passage["text"] for passage in hard_negative_context]

            # all context passages and labels: 1 for positive context and 0 for hard-negative context
            ctx_label = [1]*self.num_positives + [0]*self.num_hard_negatives #(self.num_positives if self.num_positives < len(positive_context) else len(positive_context)) + \
            # +(self.num_hard_negatives if self.num_hard_negatives < len(hard_negative_context) else len(hard_negative_context))

            # featurize context passages
            if self.embed_title:
                # embed title with positive context passages + negative context passages
                all_ctx = [tuple((title, ctx)) for title, ctx in
                           zip(positive_ctx_titles, positive_ctx_texts)] + \
                          [tuple((title, ctx)) for title, ctx in
                           zip(hard_negative_ctx_titles, hard_negative_ctx_texts)]
            else:
                all_ctx = positive_ctx_texts + hard_negative_ctx_texts

            # assign empty string tuples if hard_negative passages less than num_hard_negatives
            all_ctx += [('', '')] * ((self.num_positives + self.num_hard_negatives)-len(all_ctx))


            ctx_inputs = self.passage_tokenizer.batch_encode_plus(
                all_ctx,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len_passage,
                return_token_type_ids=True
            )


            ctx_input_ids, ctx_segment_ids_, ctx_padding_mask = ctx_inputs["input_ids"], ctx_inputs["token_type_ids"], \
                                                               ctx_inputs["attention_mask"]
            ctx_segment_ids = list(torch.zeros((len(ctx_segment_ids_), len(ctx_segment_ids_[0]))).numpy())

            # tokenize query and contexts
            tokenized_passage = [self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in ctx_input_ids]

            if len(tokenized_passage) == 0:
                logger.warning(f"The context could not be tokenized, likely because it contains a character that the context tokenizer does not recognize")
                return None

            clear_text["passages"] = positive_context + hard_negative_context
            tokenized["passages_tokens"] = tokenized_passage
            features["passage_input_ids"] = ctx_input_ids
            features["passage_segment_ids"] = ctx_segment_ids
            features["passage_attention_mask"] = ctx_padding_mask
            features["label_ids"] = ctx_label


        sample = Sample(id=None,
                        clear_text=clear_text,
                        tokenized=tokenized,
                        features=features)
        return [sample]

    def _sample_to_features(self, sample) -> dict:
        return [sample.features]

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        samples = self._dict_to_samples(dictionary, **kwargs)
        for sample in samples:
            sample.features = self._sample_to_features(sample)

        return samples


def _apply_tokenization(dictionary, tokenizer, answer_types_list=[]):
    raw_baskets = []
    dictionary = convert_qa_input_dict(dictionary)
    dictionary["qas"] = _is_impossible_to_answer_type(dictionary["qas"])
    document_text = dictionary["context"]

    document_tokenized = tokenize_with_metadata(document_text, tokenizer)
    document_start_of_word = [int(x) for x in document_tokenized["start_of_word"]]
    questions = dictionary["qas"]
    for question in questions:
        answers = []
        # For training and dev with labelled examples
        try:
            external_id = question["id"]
            question_text = question["question"]
            for answer in question["answers"]:
                if 'answer_type' in answer.keys() and answer['answer_type'] in answer_types_list:
                    answer_type = answer['answer_type']
                else:
                    if answer["text"] == "":
                        answer_type = "no_answer"
                    else:
                        answer_type = "span"
                a = {"text": answer["text"],
                     "offset": answer["answer_start"],
                     "answer_type": answer_type}
                answers.append(a)
        # For inference where samples are read in as dicts without an id or answers
        except TypeError:
            external_id = try_get(ID_NAMES, dictionary)
            question_text = question

        question_tokenized = tokenize_with_metadata(question_text, tokenizer)
        question_start_of_word = [int(x) for x in question_tokenized["start_of_word"]]

        # During inference, there is no_answer type. Also, question might be a str instead of a dict
        if type(question) == str:
            answer_type = None
        elif type(question) == dict:
            answer_type = question.get("answer_type", None)
        else:
            raise Exception("Question was neither in str nor dict format")

        raw = {"document_text": document_text,
               "document_tokens": document_tokenized["tokens"],
               "document_offsets": document_tokenized["offsets"],
               "document_start_of_word": document_start_of_word,
               "question_text": question_text,
               "question_tokens": question_tokenized["tokens"],
               "question_offsets": question_tokenized["offsets"],
               "question_start_of_word": question_start_of_word,
               "answers": answers,
               "answer_type": answer_type,
               "external_id": external_id}
        raw_baskets.append(raw)
    return raw_baskets


def _is_impossible_to_answer_type(qas):
    """ Converts questions from having an is_impossible field to having an answer_type field"""
    new_qas = []
    for q in qas:
        answer_type = "span"
        if "is_impossible" in q:
            if q["is_impossible"] == True:
                answer_type = "no_answer"
            del q["is_impossible"]
            q["answer_type"] = answer_type
        new_qas.append(q)
    return new_qas


def _check_valid_answer(sample):
    passage_text = sample.clear_text["passage_text"]
    for answer in sample.clear_text["answers"]:
        len_passage = len(passage_text)
        start = answer["start_c"]
        end = answer["end_c"]
        # Cases where the answer is not within the current passage will be turned into no answers by the featurization fn
        if start < 0 or end >= len_passage:
            continue
        answer_indices = passage_text[start: end + 1]
        answer_text = answer["text"]
        if answer_indices != answer_text:
            raise ValueError(f"""Answer using start/end indices is '{answer_indices}' while gold label text is '{answer_text}'""")

