import torch
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

from tqdm import tqdm
import multiprocessing as mp
from functools import partial

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
    create_samples_sentence_pairs,
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

from tqdm import tqdm

logger = logging.getLogger(__name__)

TOKENIZER_MAP = {"BertTokenizer": BertTokenizer}


class Processor(ABC):
    """
    Is used to generate PyTorch Datasets from input data. An implementation of this abstract class should be created
    for each new data source. Must have dataset_from_file(), dataset_from_dicts(), load(),
    load_from_file() and save() implemented in order to be compatible with the rest of the framework. The other
    functions implement our suggested pipeline structure.
    """

    subclasses = {}

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        label_list,
        metrics,
        train_filename,
        dev_filename,
        test_filename,
        dev_split,
        data_dir,
        label_dtype=torch.long,
        multiprocessing_chunk_size=1_000,
        max_processes=128,
        share_all_baskets_for_multiprocessing=False,
    ):
        """
        Initialize a generic Processor

        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param label_list: List of all unique target labels.
        :type label_list: list
        :param metrics: The metric used for evaluation, one per prediction head.
                        Choose from mcc, acc, acc_f1, pear_spear, seq_f1, f1_macro, squad.
        :type metrics: list or str
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
        :param label_dtype: The torch dtype for the labels.
        :param max_processes: maximum number of processing to use for Multiprocessing.
        :type max_processes: int
        """

        # The Multiprocessing functions in the Class are classmethods to avoid passing(and pickling) of class-objects
        # that are very large in size(eg, self.baskets). Since classmethods have access to only class attributes, all
        # objects required in Multiprocessing must be set as class attributes.
        Processor.tokenizer = tokenizer
        Processor.max_seq_len = max_seq_len
        Processor.label_list = label_list

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        self.data_dir = data_dir
        # labels
        self.label_dtype = label_dtype
        self.label_maps = []
        # multiprocessing
        self.multiprocessing_chunk_size = multiprocessing_chunk_size
        self.share_all_baskets_for_multiprocessing = (
            share_all_baskets_for_multiprocessing
        )
        self.max_processes = max_processes
        # others
        self.metrics = [metrics] if isinstance(metrics, str) else metrics

        # create label maps (one per prediction head)
        if any(isinstance(i, list) for i in label_list):
            for labels_per_head in label_list:
                map = {i: label for i, label in enumerate(labels_per_head)}
                self.label_maps.append(map)
        else:
            map = {i: label for i, label in enumerate(label_list)}
            self.label_maps.append(map)

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
        metrics,
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
        :param metrics: metrics used for each prediction head output
        :type metrics: list
        :return: An instance of the specified processor.
        """

        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(
            f"Got more parameters than needed for loading {processor_name}: {unused_args}. "
            f"Those won't be used!"
        )
        return cls.subclasses[processor_name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            metrics=metrics,
            **kwargs,
        )

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
        return cls.load(
            tokenizer=tokenizer, processor_name=config["processor"], **config
        )

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        config["tokenizer"] = self.tokenizer.__class__.__name__
        self.tokenizer.save_vocabulary(save_dir)
        # TODO make this generic to other tokenizers. We will probably want an own abstract Tokenizer
        config["lower_case"] = self.tokenizer.basic_tokenizer.do_lower_case
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

    @abc.abstractmethod
    def _file_to_dicts(self, file: str) -> [dict]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _dict_to_samples(cls, dict: dict, all_dicts=None) -> [Sample]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _sample_to_features(cls, sample: Sample) -> dict:
        raise NotImplementedError()

    def _init_baskets_from_file(self, file):
        dicts = self._file_to_dicts(file)
        dataset_name = os.path.splitext(os.path.basename(file))[0]

        self.baskets = [
            SampleBasket(raw=tr, id=f"{dataset_name}-{i}") for i, tr in enumerate(dicts)
        ]

    def _init_samples_in_baskets(self):
        chunks_to_process = int(len(self.baskets) / self.multiprocessing_chunk_size)
        num_cpus = min(mp.cpu_count(), self.max_processes, chunks_to_process) or 1

        logger.info(
            f"Got ya {num_cpus} parallel workers to fill the baskets with samples (chunksize = {self.multiprocessing_chunk_size})..."
        )

        with mp.Pool(processes=num_cpus) as p:
            with mp.Manager() as manager:
                if self.share_all_baskets_for_multiprocessing:
                    all_dicts = manager.list([b.raw for b in self.baskets])
                else:
                    all_dicts = None

                with mp.Pool(processes=num_cpus) as p:
                    samples = p.imap(
                        partial(self._multiproc_sample, all_dicts=all_dicts),
                        self.baskets,
                        chunksize=self.multiprocessing_chunk_size,
                    )

                    for s, b in tqdm(
                        zip(samples, self.baskets), total=len(self.baskets)
                    ):
                        b.samples = s

    @classmethod
    def _multiproc_sample(cls, basket, all_dicts=None):
        samples = cls._dict_to_samples(dict=basket.raw, all_dicts=all_dicts)
        for num, sample in enumerate(samples):
            sample.id = f"{basket.id}-{num}"
        return samples

    def _featurize_samples(self):
        chunks_to_process = int(len(self.baskets) / self.multiprocessing_chunk_size)
        num_cpus = min(mp.cpu_count(), self.max_processes, chunks_to_process) or 1
        logger.info(
            f"Got ya {num_cpus} parallel workers to featurize samples in baskets (chunksize = {self.multiprocessing_chunk_size}) ..."
        )
        with mp.Pool(processes=num_cpus) as p:
            all_features_gen = p.imap(
                self._multiproc_featurize,
                self.baskets,
                chunksize=self.multiprocessing_chunk_size,
            )

            for basket_features, basket in tqdm(
                zip(all_features_gen, self.baskets), total=len(self.baskets)
            ):
                for f, s in zip(basket_features, basket.samples):
                    s.features = f

    @classmethod
    def _multiproc_featurize(cls, basket):
        all_features = []
        for sample in basket.samples:
            all_features.append(cls._sample_to_features(sample=sample))
        return all_features

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

    def dataset_from_file(self, file, log_time=True):
        """
        Contains all the functionality to turn a data file into a PyTorch Dataset and a
        list of tensor names. This is used for training and evaluation.

        :param file: Name of the file containing the data.
        :type file: str
        :return: a Pytorch dataset and a list of tensor names.
        """
        if log_time:
            a = time.time()
            self._init_baskets_from_file(file)
            b = time.time()
            MlLogger.log_metrics(metrics={"t_from_file": (b - a) / 60}, step=0)
            self._init_samples_in_baskets()
            c = time.time()
            MlLogger.log_metrics(metrics={"t_init_samples": (c - b) / 60}, step=0)
            self._featurize_samples()
            d = time.time()
            MlLogger.log_metrics(metrics={"t_featurize_samples": (d - c) / 60}, step=0)
            self._log_samples(3)
        else:
            self._init_baskets_from_file(file)
            self._init_samples_in_baskets()
            self._featurize_samples()
            self._log_samples(3)
        dataset, tensor_names = self._create_dataset()
        return dataset, tensor_names

    def dataset_from_dicts(self, dicts):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This is used for inference mode.

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
        names = ["max_seq_len", "metrics", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")


#########################################
# Processors for text classification ####
#########################################
class TextClassificationProcessor(Processor):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        metrics=["acc"],
        label_dtype=torch.long,
        delimiter="\t",
        quote_char="'",
        skiprows=[0],
        columns=["text", "label"],
        **kwargs,
    ):

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.columns = columns

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metrics,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    """
    Used to handle the text classification datasets that come in tabular format (CSV, TSV, etc.)
    """

    def _file_to_dicts(self, file: str) -> [dict]:
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            columns=self.columns,
        )
        return dicts

    @classmethod
    def _dict_to_samples(cls, dict: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dict["text"], cls.tokenizer, cls.max_seq_len)
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            label_list=cls.label_list,
            max_seq_len=cls.max_seq_len,
            tokenizer=cls.tokenizer,
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
        train_filename="train.txt",
        dev_filename="dev.txt",
        test_filename="test.txt",
        dev_split=None,
        label_list=[
            "[PAD]",
            "X",
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-OTH",
            "I-OTH",
        ],
        metrics=["seq_f1"],
        label_dtype=torch.long,
        delimiter="\t",
        **kwargs,
    ):

        # Custom processor attributes
        self.delimiter = delimiter

        super(NERProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metrics,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    def _file_to_dicts(self, file: str) -> [dict]:
        dicts = read_ner_file(filename=file, sep=self.delimiter)
        return dicts

    @classmethod
    def _dict_to_samples(cls, dict: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets, which helps to map our entity tags back to original positions
        tokenized = tokenize_with_metadata(dict["text"], cls.tokenizer, cls.max_seq_len)
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        features = samples_to_features_ner(
            sample=sample,
            label_list=cls.label_list,
            max_seq_len=cls.max_seq_len,
            tokenizer=cls.tokenizer,
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
        **kwargs,
    ):
        # General Processor attributes
        label_list = [list(tokenizer.vocab), ["True", "False"]]  # labels for both heads
        metrics = ["acc", "acc"]
        label_dtype = torch.long
        chunksize = 100
        share_all_baskets_for_multiprocessing = True

        # Custom attributes
        self.delimiter = ""

        super(BertStyleLMProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metrics,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
            multiprocessing_chunk_size=chunksize,
            share_all_baskets_for_multiprocessing=share_all_baskets_for_multiprocessing,
        )

    def _file_to_dicts(self, file: str) -> list:
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter)
        return dicts

    @classmethod
    def _dict_to_samples(cls, dict, all_dicts=None):
        doc = dict["doc"]
        samples = []
        for idx in range(len(doc) - 1):
            text_a, text_b, is_next_label = get_sentence_pair(doc, all_dicts, idx)
            sample_in_clear_text = {
                "text_a": text_a,
                "text_b": text_b,
                "is_next_label": is_next_label,
            }
            tokenized = {}
            tokenized["text_a"] = tokenize_with_metadata(
                text_a, cls.tokenizer, cls.max_seq_len
            )
            tokenized["text_b"] = tokenize_with_metadata(
                text_b, cls.tokenizer, cls.max_seq_len
            )
            samples.append(
                Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized)
            )
        return samples

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        features = samples_to_features_bert_lm(
            sample=sample, max_seq_len=cls.max_seq_len, tokenizer=cls.tokenizer
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
        train_filename="train-v2.0.json",
        dev_filename="dev-v2.0.json",
        test_filename=None,
        dev_split=0,
        label_list=["start_token", "end_token"],
        metrics=["squad"],
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

        chunksize = 20

        # custom processor attributes that are accessed during multiprocessing
        # (everything you want to access in _dict_to_samples and _sample_to_features)
        SquadProcessor.doc_stride = doc_stride
        SquadProcessor.max_query_length = max_query_length

        super(SquadProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metrics,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=torch.long,  # TODO check if that is correct and needed
            multiprocessing_chunk_size=chunksize,
        )

    def dataset_from_dicts(self, dicts):
        dicts_converted = [self._convert_inference(x) for x in dicts]
        self.baskets = [
            SampleBasket(raw=tr, id="infer - {}".format(i))
            for i, tr in enumerate(dicts_converted)
        ]
        self._init_samples_in_baskets()
        self._featurize_samples()
        dataset, tensor_names = self._create_dataset()
        return dataset, tensor_names

    @classmethod
    def _convert_inference(cls, infer_dict):
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

    def _file_to_dicts(self, file: str) -> [dict]:
        dict = read_squad_file(filename=file)
        return dict

    @classmethod
    def _dict_to_samples(cls, dict: dict, **kwargs) -> [Sample]:
        # TODO split samples that are too long in this function, related to todo in self._sample_to_features
        if "paragraphs" not in dict:  # TODO change this inference mode hack
            dict = cls._convert_inference(infer_dict=dict)
        samples = create_samples_squad(entry=dict)
        for sample in samples:
            tokenized = tokenize_with_metadata(
                text=" ".join(sample.clear_text["doc_tokens"]),
                tokenizer=cls.tokenizer,
                max_seq_len=cls.max_seq_len,
            )
            sample.tokenized = tokenized

        return samples

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        # TODO, make this function return one set of features per sample
        features = sample_to_features_squad(
            sample=sample,
            tokenizer=cls.tokenizer,
            max_seq_len=cls.max_seq_len,
            doc_stride=cls.doc_stride,
            max_query_length=cls.max_query_length,
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
        label_list,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        metrics=["mse"],
        label_dtype=torch.long,
        delimiter="\t",
        quote_char="'",
        skiprows=[0],
        columns=["text", "label"],
        scaler_mean=None,
        scaler_scale=None,
        **kwargs,
    ):

        # Custom processor attributes
        self.label_list = [scaler_mean, scaler_scale]
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.columns = columns

        super(RegressionProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metrics,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a pkl file for the scaler and
        a json file containing all the information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        config["tokenizer"] = self.tokenizer.__class__.__name__
        self.tokenizer.save_vocabulary(save_dir)
        # TODO make this generic to other tokenizers. We will probably want an own abstract Tokenizer
        config["lower_case"] = self.tokenizer.basic_tokenizer.do_lower_case
        config["max_seq_len"] = self.max_seq_len
        config["processor"] = self.__class__.__name__
        config["scaler_mean"] = self.label_list[0]
        config["scaler_scale"] = self.label_list[1]
        output_config_file = os.path.join(save_dir, "processor_config.json")
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def _file_to_dicts(self, file: str) -> [dict]:
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            columns=self.columns,
        )
        return dicts

    @classmethod
    def _dict_to_samples(cls, dict: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dict["text"], cls.tokenizer, cls.max_seq_len)
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            label_list=cls.label_list,
            max_seq_len=cls.max_seq_len,
            tokenizer=cls.tokenizer,
        )
        return features

    def _featurize_samples(self):
        chunks_to_process = int(len(self.baskets) / self.multiprocessing_chunk_size)
        num_cpus = min(mp.cpu_count(), self.max_processes, chunks_to_process) or 1
        logger.info(
            f"Got ya {num_cpus} parallel workers to featurize samples in baskets (chunksize = {self.multiprocessing_chunk_size}) ..."
        )

        try:
            if "train" in self.baskets[0].id:
                train_labels = []
                for basket in self.baskets:
                    for sample in basket.samples:
                        train_labels.append(sample.clear_text["label"])
                scaler = StandardScaler()
                scaler.fit(np.reshape(train_labels, (-1, 1)))
                self.label_list = [scaler.mean_.item(), scaler.scale_.item()]
                # Create label_maps because featurize is called after Processor instantiation
                self.label_maps = [{0:scaler.mean_.item(), 1:scaler.scale_.item()}]

        except Exception as e:
            logger.warning(f"Baskets not found: {e}")

        with mp.Pool(processes=num_cpus) as p:
            all_features_gen = p.imap(
                self._multiproc_featurize,
                self.baskets,
                chunksize=self.multiprocessing_chunk_size,
            )

            for basket_features, basket in tqdm(
                zip(all_features_gen, self.baskets), total=len(self.baskets)
            ):
                for f, s in zip(basket_features, basket.samples):
                    # Samples don't have labels during Inference mode
                    if "label" in s.clear_text:
                        label = s.clear_text["label"]
                        scaled_label = (label - self.label_list[0]) / self.label_list[1]
                        f[0]["label_ids"] = scaled_label
                    s.features = f