import os
import abc
from abc import ABC
import random
import logging
import json
import time
import inspect
from inspect import signature
import numpy as np
from sklearn.preprocessing import StandardScaler

from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.input_features import (
    samples_to_features_ner,
    samples_to_features_bert_lm,
    sample_to_features_text,
    sample_to_features_squad,
)
from farm.data_handler.samples import (
    Sample,
    SampleBasket,
    create_samples_squad,
)
from farm.data_handler.utils import (
    read_tsv,
    read_docs_from_txt,
    read_ner_file,
    read_squad_file,
    is_json,
)
from farm.modeling.tokenization import BertTokenizer, tokenize_with_metadata
from farm.utils import MLFlowLogger as MlLogger
from farm.data_handler.samples import get_sentence_pair

logger = logging.getLogger(__name__)

TOKENIZER_MAP = {"BertTokenizer": BertTokenizer}


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
        tasks={}
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
        """

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        self.data_dir = data_dir

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
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
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
        processor_config_file = os.path.join(load_dir, "processor_config.json")
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = TOKENIZER_MAP[config["tokenizer"]].from_pretrained(
            load_dir,
            do_lower_case=config["lower_case"],
            never_split_chars=config.get("never_split_chars", None),
        )
        # add custom vocab to tokenizer if available
        if os.path.exists(os.path.join(load_dir, "custom_vocab.txt")):
            tokenizer.add_custom_vocab(os.path.join(load_dir, "custom_vocab.txt"))
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

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
        self.tokenizer.save_vocabulary(save_dir)
        # TODO make this generic to other tokenizers. We will probably want an own abstract Tokenizer
        config["lower_case"] = self.tokenizer.basic_tokenizer.do_lower_case
        config["never_split_chars"] = self.tokenizer.basic_tokenizer.never_split_chars
        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = os.path.join(save_dir, "processor_config.json")
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
                config[key] = value
        return config

    def add_task(self, name,  metric, label_list, label_column_name=None, label_name=None, task_type=None):
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

    def _init_baskets_from_file(self, file):
        dicts = self.file_to_dicts(file)
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        baskets = [
            SampleBasket(raw=tr, id=f"{dataset_name}-{i}") for i, tr in enumerate(dicts)
        ]
        return baskets

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
            basket.samples = self._dict_to_samples(dictionary=basket.raw, all_dicts=all_dicts)
            for num, sample in enumerate(basket.samples):
                 sample.id = f"{basket.id}-{num}"

    def _featurize_samples(self):
        for basket in self.baskets:
            for sample in basket.samples:
                sample.features = self._sample_to_features(sample=sample)

    def _create_dataset(self, keep_baskets=False):
        features_flat = []
        for basket in self.baskets:
            for sample in basket.samples:
                features_flat.extend(sample.features)
        if not keep_baskets:
            # free up some RAM, we don't need baskets from here on
            self.baskets = None
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names

    # def dataset_from_file(self, file, log_time=True):
    #     """
    #     Contains all the functionality to turn a data file into a PyTorch Dataset and a
    #     list of tensor names. This is used for training and evaluation.
    #
    #     :param file: Name of the file containing the data.
    #     :type file: str
    #     :return: a Pytorch dataset and a list of tensor names.
    #     """
    #     if log_time:
    #         a = time.time()
    #         self._init_baskets_from_file(file)
    #         b = time.time()
    #         MlLogger.log_metrics(metrics={"t_from_file": (b - a) / 60}, step=0)
    #         self._init_samples_in_baskets()
    #         c = time.time()
    #         MlLogger.log_metrics(metrics={"t_init_samples": (c - b) / 60}, step=0)
    #         self._featurize_samples()
    #         d = time.time()
    #         MlLogger.log_metrics(metrics={"t_featurize_samples": (d - c) / 60}, step=0)
    #         self._log_samples(3)
    #     else:
    #         self._init_baskets_from_file(file)
    #         self._init_samples_in_baskets()
    #         self._featurize_samples()
    #         self._log_samples(3)
    #     dataset, tensor_names = self._create_dataset()
    #     return dataset, tensor_names

    #TODO remove useless rest_api_schema flag after refactoring squad processing
    def dataset_from_dicts(self, dicts, index=None, rest_api_schema=False):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This can be used for inference mode.

        :param dicts: List of dictionaries where each contains the data of one input sample.
        :type dicts: list of dicts
        :return: a Pytorch dataset and a list of tensor names.
        """
        self.baskets = [
            SampleBasket(raw=tr, id="infer - {}".format(i))
            for i, tr in enumerate(dicts)
        ]
        self._init_samples_in_baskets()
        self._featurize_samples()
        if index == 0:
            self._log_samples(3)
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
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")


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
        **kwargs,
    ):
        #TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
        )
        #TODO raise info when no task is added due to missing "metric" or "labels" arg
        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(name="text_classification",
                          metric=metric,
                          label_list=label_list,
                          label_column_name=label_column_name,
                          task_type=task_type)

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            rename_columns=column_mapping,
            header=self.header
            )

        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer, self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


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
            tasks={}
        )

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Overwriting method from parent class to **always** load the InferenceProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an InferenceProcessor
        """
        # read config
        processor_config_file = os.path.join(load_dir, "processor_config.json")
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = TOKENIZER_MAP[config["tokenizer"]].from_pretrained(
            load_dir,
            do_lower_case=config["lower_case"],
            never_split_chars=config.get("never_split_chars", None),
        )
        # add custom vocab to tokenizer if available
        if os.path.exists(os.path.join(load_dir, "custom_vocab.txt")):
            tokenizer.add_custom_vocab(os.path.join(load_dir, "custom_vocab.txt"))
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
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer, self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features

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
        **kwargs,
    ):

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
            tasks={}
        )

        if metric and label_list:
            self.add_task("ner", metric, label_list)

    def file_to_dicts(self, file: str) -> [dict]:
        dicts = read_ner_file(filename=file, sep=self.delimiter)
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets, which helps to map our entity tags back to original positions
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer, self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_ner(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


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
        max_docs=None,
        **kwargs,
    ):

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
            tasks={}
        )

        self.next_sent_pred = next_sent_pred

        self.add_task("lm", "acc", list(self.tokenizer.vocab))
        if self.next_sent_pred:
            self.add_task("nextsentence", "acc", ["False", "True"])


    def file_to_dicts(self, file: str) -> list:
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter, max_docs=self.max_docs)
        return dicts

    def _dict_to_samples(self, dictionary, all_dicts=None):
        doc = dictionary["doc"]
        samples = []
        for idx in range(len(doc) - 1):
            text_a, text_b, is_next_label = get_sentence_pair(doc, all_dicts, idx)
            sample_in_clear_text = {
                "text_a": text_a,
                "text_b": text_b,
                "nextsentence_label": is_next_label,
            }
            tokenized = {}
            tokenized["text_a"] = tokenize_with_metadata(
                text_a, self.tokenizer, self.max_seq_len
            )
            tokenized["text_b"] = tokenize_with_metadata(
                text_b, self.tokenizer, self.max_seq_len
            )
            samples.append(
                Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized)
            )
        return samples

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_bert_lm(
            sample=sample, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer,
            next_sent_pred=self.next_sent_pred
        )
        return features


#########################################
# SQUAD 2.0 Processor ####
#########################################
class SquadProcessor(Processor):
    """ Used to handle the SQuAD dataset"""

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        labels=None,
        metric=None,
        train_filename="train-v2.0.json",
        dev_filename="dev-v2.0.json",
        test_filename=None,
        dev_split=0,
        doc_stride=128,
        max_query_length=64,
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
        :type data_dir: str
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :type data_dir: str
        :param doc_stride: When the document containing the answer is too long it gets split into part, strided by doc_stride
        :type doc_stride: int
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :type max_query_length: int
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        self.target = "classification"
        self.ph_output_type = "per_token_squad"

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

        super(SquadProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
        )

        if metric and labels:
            self.add_task("question_answering", metric, labels)

    def dataset_from_dicts(self, dicts, index=None, rest_api_schema=False):
        if rest_api_schema:
            dicts = [self._convert_rest_api_dict(x) for x in dicts]
        if rest_api_schema:
            id_prefix = "infer"
        else:
            id_prefix = "train"
        self.baskets = [
            SampleBasket(raw=tr, id=f"{id_prefix}-{i}")
            for i, tr in enumerate(dicts)
        ]
        self._init_samples_in_baskets()
        self._featurize_samples()
        if index == 0:
            self._log_samples(3)
        dataset, tensor_names = self._create_dataset()
        return dataset, tensor_names

    def _convert_rest_api_dict(self, infer_dict):
        # convert input coming from inferencer to SQuAD format
        converted = {}
        converted["paragraphs"] = [
            {
                "qas": [
                    {
                        "question": infer_dict.get("questions", ["Missing?"])[0],
                        "id": "unusedID",
                    }
                ],
                "context": infer_dict.get("text", "Missing!"),
            }
        ]
        return converted

    def file_to_dicts(self, file: str) -> [dict]:
        dict = read_squad_file(filename=file)
        return dict

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        if "paragraphs" not in dictionary:  # TODO change this inference mode hack
            dictionary = self._convert_rest_api_dict(infer_dict=dictionary)
        samples = create_samples_squad(entry=dictionary)
        for sample in samples:
            tokenized = tokenize_with_metadata(
                text=" ".join(sample.clear_text["doc_tokens"]),
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )
            sample.tokenized = tokenized

        return samples

    def _sample_to_features(self, sample) -> dict:
        # TODO, make this function return one set of features per sample
        features = sample_to_features_squad(
            sample=sample,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            tasks=self.tasks
        )
        return features


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
        **kwargs,
    ):

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
        )

        self.add_task(name="regression", metric="mse", label_list= [scaler_mean, scaler_scale], label_column_name=label_column_name, task_type="regression", label_name=label_name)


    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
        dicts = read_tsv(
            rename_columns=column_mapping,
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
        )

        # collect all labels and compute scaling stats
        train_labels = []
        for d in dicts:
            train_labels.append(float(d[self.tasks["regression"]["label_name"]]))
        scaler = StandardScaler()
        scaler.fit(np.reshape(train_labels, (-1, 1)))
        # add to label list in regression task
        self.tasks["regression"]["label_list"] = [scaler.mean_.item(), scaler.scale_.item()]

        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer, self.max_seq_len)
        # Samples don't have labels during Inference mode
        if "label" in dictionary:
            label = float(dictionary["label"])
            scaled_label = (label - self.tasks["regression"]["label_list"][0]) / self.tasks["regression"]["label_list"][1]
            dictionary["label"] = scaled_label
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer
        )
        return features