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

import numpy as np
from sklearn.preprocessing import StandardScaler

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
    read_tsv,
    read_tsv_sentence_pair,
    read_docs_from_txt,
    read_ner_file,
    read_squad_file,
    read_jsonl,
    is_json,
    get_sentence_pair,
    split_with_metadata,
    convert_qa_input_dict,
    get_sequence_pair,
    join_sentences
)

from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata, truncate_sequences
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

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
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
                    logger.error(f"Error message: {e}")

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

    def get_added_tokens(self):
        dictionary = self.tokenizer.added_tokens_encoder
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
            next_sent_pred=self.next_sent_pred
        )
        return features

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

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        """ Overwrites the method from the base class since Question Answering processing is quite different.
        This method allows for documents and questions to be tokenized earlier. Then SampleBaskets are initialized
        with one document and one question. """

        dicts = [convert_qa_input_dict(x) for x in dicts]
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
                                         sp_toks_mid=self.sp_toks_mid)
        return features

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
        # Turns a NQ dictionaries into a SQuAD style dictionaries
        if not self.inference:
            dictionary = self._prepare_dict(dictionary=dictionary)

        dictionary_tokenized = _apply_tokenization(dictionary, self.tokenizer)[0]
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
                                         answer_type_list=self.answer_type_list)
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


def _apply_tokenization(dictionary, tokenizer):
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

