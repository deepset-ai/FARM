import abc
import inspect
import json
import logging
import os
import random
from abc import ABC
from inspect import signature
from pathlib import Path

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
    create_samples_qa
)
from farm.data_handler.utils import (
    read_tsv,
    read_tsv_sentence_pair,
    read_docs_from_txt,
    read_ner_file,
    read_squad_file,
    is_json,
    get_sentence_pair,
    split_with_metadata
)
from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata, truncate_sequences
from farm.utils import MLFlowLogger as MlLogger


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
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
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
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"],
                               label_column_name=task["label_column_name"], task_type=task["task_type"])

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
        self.tokenizer.save_pretrained(save_dir)
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
        dataset_name = file.stem
        baskets = [
            SampleBasket(raw=tr, id=f"{dataset_name}-{i}") for i, tr in enumerate(dicts)
        ]
        return baskets

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
            try:
                basket.samples = self._dict_to_samples(dictionary=basket.raw, all_dicts=all_dicts)
                for num, sample in enumerate(basket.samples):
                     sample.id = f"{basket.id}-{num}"
            except:
                logger.error(f"Could not create sample(s) from this dict: \n {basket.raw}")
                raise

    def _featurize_samples(self):
        for basket in self.baskets:
            for sample in basket.samples:
                try:
                    sample.features = self._sample_to_features(sample=sample)
                except:
                    logger.error(f"Could not convert this sample to features: \n {sample}")
                    raise

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

    def dataset_from_dicts(self, dicts, indices=None, rest_api_schema=False, return_baskets = False):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This can be used for inference mode.

        :param dicts: List of dictionaries where each contains the data of one input sample.
        :type dicts: list of dicts
        :return: a Pytorch dataset and a list of tensor names.
        """
        if rest_api_schema:
            id_prefix = "infer"
        else:
            id_prefix = "train"
        # We need to add the index (coming from multiprocessing chunks) to have a unique basket ID
        if indices:
            self.baskets = [
                SampleBasket(raw=tr, id=f"{id_prefix}-{index}")
                for (tr, index) in zip(dicts, indices)
            ]
        else:
            self.baskets = [
                SampleBasket(raw=tr, id=f"{id_prefix}-{i}")
                for (i, tr) in enumerate(dicts)
            ]
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
        proxies=None,
        max_samples=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
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
                          task_type=task_type)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
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
        tokenized = {"tokens": tokenized_a["tokens"],
                     "tokens_b": tokenized_b["tokens"]}
        tokenized["tokens"], tokenized["tokens_b"], _ = truncate_sequences(seq_a=tokenized["tokens"],
                                                                           seq_b=tokenized["tokens_b"],
                                                                           tokenizer=self.tokenizer,
                                                                           max_seq_len=self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]


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
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
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
        proxies=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
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
        :param max_docs: maximum number of documents to include from input dataset
        :type max_docs: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
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
        added_tokens = self.get_added_tokens()
        self.add_task("lm", "acc", list(self.tokenizer.vocab) + added_tokens)
        if self.next_sent_pred:
            self.add_task("nextsentence", "acc", ["False", "True"])

    def get_added_tokens(self):
        dictionary = self.tokenizer.added_tokens_encoder
        sorted_tuples = sorted(dictionary.items(), key=lambda x: x[0])
        return [x[1] for x in sorted_tuples]

    def file_to_dicts(self, file: str) -> list:
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter, max_docs=self.max_docs, proxies=self.proxies)
        return dicts

    def _dict_to_samples(self, dictionary, all_dicts=None):
        assert len(all_dicts) > 1, "Need at least 2 documents to sample random sentences from"
        doc = dictionary["doc"]
        samples = []

        # create one sample for each sentence in the doc (except for the very last -> "nextSentence" is impossible)
        for idx in range(len(doc) - 1):
            tokenized = {}
            if self.next_sent_pred:
                text_a, text_b, is_next_label = get_sentence_pair(doc, all_dicts, idx)
                sample_in_clear_text = {
                    "text_a": text_a,
                    "text_b": text_b,
                    "nextsentence_label": is_next_label,
                }
                # tokenize
                tokenized["text_a"] = tokenize_with_metadata(
                    text_a, self.tokenizer
                )
                tokenized["text_b"] = tokenize_with_metadata(
                    text_b, self.tokenizer
                )
                # truncate to max_seq_len
                for seq_name in ["tokens", "offsets", "start_of_word"]:
                    tokenized["text_a"][seq_name], tokenized["text_b"][seq_name], _ = truncate_sequences(
                        seq_a=tokenized["text_a"][seq_name],
                        seq_b=tokenized["text_b"][seq_name],
                        tokenizer=self.tokenizer,
                        max_seq_len=self.max_seq_len)
                samples.append(Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized))
            # if we don't do next sentence prediction, we should feed in a single sentence
            else:
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
                # truncate to max_seq_len
                for seq_name in ["tokens", "offsets", "start_of_word"]:
                    tokenized["text_a"][seq_name], _, _ = truncate_sequences(
                        seq_a=tokenized["text_a"][seq_name],
                        seq_b=None,
                        tokenizer=self.tokenizer,
                        max_seq_len=self.max_seq_len)
                samples.append(Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized))
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
        label_list=None,
        metric="squad",
        train_filename=Path("train-v2.0.json"),
        dev_filename=Path("dev-v2.0.json"),
        test_filename=None,
        dev_split=0,
        doc_stride=128,
        max_query_length=64,
        proxies=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, can be "squad" or "squad_top_recall"
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
            proxies=proxies
        )

        if metric and label_list:
            self.add_task("question_answering", metric, label_list)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def dataset_from_dicts(self, dicts, indices=None, rest_api_schema=False, return_baskets=False):
        """ Overwrites the method from the base class since Question Answering processing is quite different.
        This method allows for documents and questions to be tokenized earlier. Then SampleBaskets are initialized
        with one document and one question. """

        if rest_api_schema:
            dicts = [self._convert_rest_api_dict(x) for x in dicts]
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
        # Perform tokenization on documents and questions resulting in a nested list of doc-question pairs
        dicts_tokenized = [self.apply_tokenization(d) for d in dicts]

        baskets = []
        for index, document in zip(indices, dicts_tokenized):
            for q_idx, raw in enumerate(document):
                # In case of Question Answering the external ID is used for document IDs
                basket = SampleBasket(raw=raw, id=f"{index}-{q_idx}", external_id=raw.get("document_id",None))
                baskets.append(basket)
        return baskets


    def apply_tokenization(self, dictionary):
        """ This performs tokenization on all documents and questions. The result is a list (unnested)
        where each entry is a dictionary for one document-question pair (potentially mutliple answers). """

        raw_baskets = []
        if "text" in dictionary and "context" not in dictionary:
            raise Exception("It seems that your input is in rest API format. Try setting rest_api_schema=True "
                            "when calling inference from dicts")
        document_text = dictionary["context"]
        document_id = dictionary.get("document_id",None)

        document_tokenized = tokenize_with_metadata(document_text, self.tokenizer)
        document_start_of_word = [int(x) for x in document_tokenized["start_of_word"]]
        questions = dictionary["qas"]
        for question in questions:
            answers = []
            # For training and dev where labelled samples are read in from a SQuAD style file
            try:
                squad_id = question["id"]
                question_text = question["question"]
                for answer in question["answers"]:
                    a = {"text": answer["text"],
                         "offset": answer["answer_start"]}
                    answers.append(a)
            # For inference where samples are read in as dicts without an id or answers
            except TypeError:
                squad_id = None
                question_text = question
            question_tokenized = tokenize_with_metadata(question_text, self.tokenizer)
            question_start_of_word = [int(x) for x in question_tokenized["start_of_word"]]

            if "is_impossible" not in question:
                is_impossible = False
            else:
                is_impossible = question["is_impossible"]
            raw = {"document_text": document_text,
                   "document_tokens": document_tokenized["tokens"],
                   "document_offsets": document_tokenized["offsets"],
                   "document_start_of_word": document_start_of_word,
                   "document_id": document_id,
                   "question_text": question_text,
                   "question_tokens": question_tokenized["tokens"],
                   "question_offsets": question_tokenized["offsets"],
                   "question_start_of_word": question_start_of_word,
                   "answers": answers,
                   "is_impossible": is_impossible,
                   "squad_id": squad_id}
            raw_baskets.append(raw)
        return raw_baskets

    def _convert_rest_api_dict(self, infer_dict):
        # converts dicts from inference mode to data structure used in FARM
        questions = infer_dict.get("questions", None)
        text = infer_dict.get("text", None)
        document_id = infer_dict.get("document_id", None)
        qas = [{"question": q,
                "id": i,
                "answers": [],
                "is_impossible": False} for i, q in enumerate(questions)]
        converted = {"qas": qas,
                     "context": text,
                     "document_id":document_id}
        return converted

    def file_to_dicts(self, file: str) -> [dict]:
        nested_dicts = read_squad_file(filename=file)
        dicts = [y for x in nested_dicts for y in x["paragraphs"]]
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        n_special_tokens = self.tokenizer.num_added_tokens(pair=True)
        samples = create_samples_squad(dictionary=dictionary,
                                       max_query_len=self.max_query_length,
                                       max_seq_len=self.max_seq_len,
                                       doc_stride=self.doc_stride,
                                       n_special_tokens=n_special_tokens)
        return samples

    def _sample_to_features(self, sample) -> dict:
        # TODO, make this function return one set of features per sample
        features = sample_to_features_squad(sample=sample,
                                            tokenizer=self.tokenizer,
                                            max_seq_len=self.max_seq_len)
        return features

class NaturalQuestionsProcessor(Processor):
    def __init__(self,
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
                 **kwargs):
            """
            :param tokenizer: Used to split a sentence (str) into tokens.
            :param max_seq_len: Samples are truncated after this many tokens.
            :type max_seq_len: int
            :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
            :type data_dir: str
            :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
            :type label_list: list
            :param metric: name of metric that shall be used for evaluation
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

            self.doc_stride = doc_stride
            self.max_query_length = max_query_length

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

            if metric and label_list:
                self.add_task("question_answering", metric, label_list)
            else:
                logger.info(
                    "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                    "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        dicts = [json.loads(l) for l in open(file)]
        return dicts

    def _dict_to_samples(self, dictionary: dict, all_dicts=None) -> [Sample]:
        """
            This method will split question-document pairs from the SampleBasket into question-passage pairs which will
        each form one sample. The "t" and "c" in variables stand for token and character respectively.
        """
        dictionary = self.prepare_dict(dictionary)
        dictionary_tokenized = self.apply_tokenization(dictionary)[0]
        n_special_tokens = self.tokenizer.num_added_tokens(pair=True)
        samples = create_samples_qa(dictionary_tokenized,
                                    self.max_query_length,
                                    self.max_seq_len,
                                    self.doc_stride,
                                    n_special_tokens)
        return samples

    def prepare_dict(self, dictionary):
        converted_answers = []
        doc_text = dictionary["document_text"]
        doc_tokens, tok_to_ch = split_with_metadata(doc_text)
        for annotation in dictionary["annotations"]:
            sa_text, sa_start_c = self.unify_short_answers(annotation["short_answers"], doc_text, tok_to_ch)
            la_text, la_start_c = self.retrieve_long_answer(annotation["long_answer"]["start_token"],
                                                            annotation["long_answer"]["end_token"],
                                                            tok_to_ch,
                                                            doc_text)
            text, start_c = self.choose_span(sa_text, sa_start_c, la_text, la_start_c)
            converted_answers.append({"text": text,
                                      "answer_start": start_c})
        if len(dictionary["annotations"]) == 0:
            answer_type = "is_impossible"
        else:
            answer_type = dictionary["annotations"][0]["yes_no_answer"]
            answer_type = answer_type.lower()
            if answer_type == "none":
                answer_type = "span"
        converted = {"id": dictionary["example_id"],
                     "context": doc_text,
                     "qas": [{"question": dictionary["question_text"],
                              "id": dictionary["example_id"],
                              "answers": converted_answers,
                              "answer_type": answer_type}]}
        return converted

    def retrieve_long_answer(self, start_t, end_t, tok_to_ch, doc_text):
        start_c, end_c = self.convert_tok_to_ch(start_t, end_t, tok_to_ch, doc_text)
        text = doc_text[start_c: end_c]
        return text, start_c

    @staticmethod
    def choose_span(sa_text, sa_start_c, la_text, la_start_c):
        if sa_text:
            return sa_text, sa_start_c
        elif la_text:
            return la_text, la_start_c
        else:
            return "", -1

    def unify_short_answers(self, short_answers, doc_text, tok_to_ch):
        if not short_answers:
            return "", -1
        short_answer_idxs = []
        # TODO write comment explaining this
        for short_answer in short_answers:
            short_answer_idxs.append(short_answer["start_token"])
            short_answer_idxs.append(short_answer["end_token"])
        answer_start_t = min(short_answer_idxs)
        answer_end_t = max(short_answer_idxs)
        answer_start_c, answer_end_c = self.convert_tok_to_ch(answer_start_t, answer_end_t, tok_to_ch, doc_text)
        answer_text = doc_text[answer_start_c: answer_end_c]
        assert answer_text == " ".join(doc_text.split()[answer_start_t: answer_end_t])
        return answer_text, answer_start_c

    @staticmethod
    def convert_tok_to_ch(start_t, end_t, tok_to_ch, doc_text):
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

    def apply_tokenization(self, dictionary):
        """ This performs tokenization on all documents and questions. The result is a list (unnested)
        where each entry is a dictionary for one document-question pair (potentially mutliple answers). """

        raw_baskets = []
        if "text" in dictionary and "context" not in dictionary:
            raise Exception("It seems that your input is in rest API format. Try setting rest_api_schema=True "
                            "when calling inference from dicts")
        document_text = dictionary["context"]
        document_id = dictionary.get("document_id",None)

        document_tokenized = tokenize_with_metadata(document_text, self.tokenizer)
        document_start_of_word = [int(x) for x in document_tokenized["start_of_word"]]
        questions = dictionary["qas"]
        for question in questions:
            answers = []
            # For training and dev where labelled samples are read in from a SQuAD style file
            try:
                squad_id = question["id"]
                question_text = question["question"]
                for answer in question["answers"]:
                    a = {"text": answer["text"],
                         "offset": answer["answer_start"],
                         "answer_type": question["answer_type"]}
                    answers.append(a)
            # For inference where samples are read in as dicts without an id or answers
            except TypeError:
                squad_id = None
                question_text = question
            question_tokenized = tokenize_with_metadata(question_text, self.tokenizer)
            question_start_of_word = [int(x) for x in question_tokenized["start_of_word"]]

            # TODO: Get rid of is_impossible key for NQ and SQUAD
            if "is_impossible" not in question:
                is_impossible = False
            else:
                is_impossible = question["is_impossible"]

            raw = {"document_text": document_text,
                   "document_tokens": document_tokenized["tokens"],
                   "document_offsets": document_tokenized["offsets"],
                   "document_start_of_word": document_start_of_word,
                   "document_id": document_id,
                   "question_text": question_text,
                   "question_tokens": question_tokenized["tokens"],
                   "question_offsets": question_tokenized["offsets"],
                   "question_start_of_word": question_start_of_word,
                   "answers": answers,
                   "is_impossible": is_impossible,
                   "squad_id": squad_id}
            raw_baskets.append(raw)
        return raw_baskets

    def _sample_to_features(self, sample: Sample) -> dict:
        answer_type_list = ["is_impossible", "span", "yes", "no"]
        features = sample_to_features_squad(sample=sample,
                                            tokenizer=self.tokenizer,
                                            max_seq_len=self.max_seq_len,
                                            answer_type_list=answer_type_list)
        return features




#     def create_samples_nq(cls, dictionary):
#         # Initialize some basic variables
#         # question_tokens = dictionary["question_tokens"][:max_query_len]
#         # question_len_t = len(question_tokens)
#         # question_offsets = dictionary["question_offsets"]
#         samples = []
#         n_special_tokens = cls.tokenizer.num_added_tokens(pair=True)
#
#         doc_text = dictionary["document_text"]
#         document_tokenized = tokenize_with_metadata(doc_text, cls.tokenizer)
#         doc_tokens = document_tokenized["tokens"]
#         doc_start_of_word = [int(x) for x in document_tokenized["start_of_word"]]
#         doc_offsets = document_tokenized["offsets"]
#
#         question_text = dictionary["question_text"]
#         question_tokenized = tokenize_with_metadata(question_text, cls.tokenizer)
#         question_tokens = question_tokenized["tokens"]
#         question_len_t = len(question_tokenized["tokens"])
#         question_offsets = document_tokenized["offsets"]
#
#         # Calculate the number of tokens that can be reserved for the passage. This is calculated by considering
#         # the max_seq_len, the number of tokens in the question and the number of special tokens that will be added
#         # when the question and passage are joined (e.g. [CLS] and [SEP])
#         passage_len_t = cls.max_seq_len - question_len_t - n_special_tokens
#
#         # Perform chunking of document into passages. The sliding window moves in steps of doc_stride.
#         # passage_spans is a list of dictionaries where each defines the start and end of each passage
#         # on both token and character level
#         passage_spans = chunk_into_passages(doc_offsets,
#                                             cls.doc_stride,
#                                             passage_len_t,
#                                             doc_text)
#         for passage_span in passage_spans:
#             # Unpack each variable in the dictionary. The "_t" and "_c" indicate
#             # whether the index is on the token or character level
#             passage_start_t = passage_span["passage_start_t"]
#             passage_end_t = passage_span["passage_end_t"]
#             passage_start_c = passage_span["passage_start_c"]
#             passage_end_c = passage_span["passage_end_c"]
#             passage_id = passage_span["passage_id"]
#
#             # passage_offsets will be relative to the start of the passage (i.e. they will start at 0)
#             # TODO: Is passage offsets actually needed? At this point, maybe we only care about token level
#             passage_offsets = doc_offsets[passage_start_t: passage_end_t]
#             passage_start_of_word = doc_start_of_word[passage_start_t: passage_end_t]
#             passage_offsets = [x - passage_offsets[0] for x in passage_offsets]
#             passage_tokens = doc_tokens[passage_start_t: passage_end_t]
#             passage_text = dictionary["document_text"][passage_start_c: passage_end_c]
#
#             # Deal with the potentially many answers (e.g. Squad dev set)
#             answers_clear, answers_tokenized = process_answers_nq(dictionary["annotations"],
#                                                                   doc_offsets,
#                                                                   passage_start_c,
#                                                                   passage_start_t)
#
#             clear_text = {"passage_text": passage_text,
#                           "question_text": dictionary["question_text"],
#                           "passage_id": passage_id,
#                           "answers": answers_clear,
#                           "is_impossible": dictionary["is_impossible"]}
#             tokenized = {"passage_start_t": passage_start_t,
#                          "passage_tokens": passage_tokens,
#                          "passage_offsets": passage_offsets,
#                          "passage_start_of_word": passage_start_of_word,
#                          "question_tokens": question_tokens,
#                          "question_offsets": question_offsets,
#                          "question_start_of_word": dictionary["question_start_of_word"][:cls.max_query_len],
#                          "answers": answers_tokenized}
#             samples.append(Sample(id=passage_id,
#                                   clear_text=clear_text,
#                                   tokenized=tokenized))
#         return samples
#
#     def _sample_to_features(cls, sample: Sample) -> dict:
#         raise NotImplementedError
#
# def process_answers_nq(answers, doc_offsets, passage_start_c, passage_start_t):
#     answers_clear = []
#     answers_tokenized = []
#     for answer in answers:
#         # NQ samples can have multiple short answers - we follow basic this implementation
#         # (https://arxiv.org/pdf/1901.08634.pdf)
#         #       "we set the start and end target indices to point to the smallest span containing all
#         #       the annotated short answer spans"
#         all_indices = []
#         for sa in answer["short_answers"]:
#             all_indices.append(sa["start_token"])
#             all_indices.append(sa["end_token"])
#         start_index = min(all_indices)
#         end_index = max(all_indices)
#         print()
#
#
#
#
#
#
#
#
#
#
#         # This section calculates start and end relative to document
#         answer_text = answer["annotations"]
#         answer_len_c = len(answer_text)
#         answer_start_c = answer["offset"]
#         answer_end_c = answer_start_c + answer_len_c - 1
#         answer_start_t = offset_to_token_idx(doc_offsets, answer_start_c)
#         answer_end_t = offset_to_token_idx(doc_offsets, answer_end_c)
#
#         # TODO: Perform check that answer can be recovered from document?
#
#         # This section converts start and end so that they are relative to the passage
#         # TODO: Is this actually necessary on character level?
#         answer_start_c -= passage_start_c
#         answer_end_c -= passage_start_c
#         answer_start_t -= passage_start_t
#         answer_end_t -= passage_start_t
#
#         curr_answer_clear = {"text": answer_text,
#                              "start_c": answer_start_c,
#                              "end_c": answer_end_c}
#         curr_answer_tokenized = {"start_t": answer_start_t,
#                                  "end_t": answer_end_t}
#
#         answers_clear.append(curr_answer_clear)
#         answers_tokenized.append(curr_answer_tokenized)
#
#



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
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
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
        :param label_name: name for the internal label variable in FARM (only needed to adjust in rare cases)
        :type label_name: str
        :param scaler_mean: Value to substract from the label for normalization
        :type scaler_mean: float
        :param scaler_scale: Value to divide the label by for normalization
        :type scaler_scale: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
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
        self.add_task(name="regression", metric="mse", label_list=[scaler_mean, scaler_scale], label_column_name=label_column_name, task_type="regression", label_name=label_name)

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
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
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None,
                                                           tokenizer=self.tokenizer,
                                                           max_seq_len=self.max_seq_len)
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