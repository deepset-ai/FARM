import torch
import os
import abc
from abc import ABC
import random
import logging
import json

from farm.modeling.tokenization import BertTokenizer

from farm.modeling.tokenization import tokenize_with_metadata

from farm.data_handler.utils import (
    read_tsv,
    read_docs_from_txt,
    read_ner_file,
    read_squad_file,
)
from farm.file_utils import create_folder
from farm.data_handler.samples import (
    Sample,
    SampleBasket,
    create_samples_sentence_pairs,
    create_samples_squad,
)
from farm.data_handler.input_features import (
    samples_to_features_ner,
    samples_to_features_bert_lm,
    sample_to_features_text,
    sample_to_features_squad,
)
from farm.data_handler.dataset import convert_features_to_dataset
from farm.utils import MLFlowLogger as MlLogger


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

        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_list = label_list
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        self.data_dir = data_dir
        self.label_dtype = label_dtype
        self.label_maps = []

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
    def load(cls, processor_name, data_dir, tokenizer, max_seq_len):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :type processor_name: str
        :param data_dir: Directory where data files are located.
        :type data_dir: str
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :type max_seq_len: int
        :return: An instance of the specified processor.
        """
        return cls.subclasses[processor_name](
            data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len
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
            load_dir, do_lower_case=config["lower_case"], never_split_chars=config.get("never_split_chars")
        )
        # add custom vocab to tokenizer if available
        if os.path.exists(os.path.join(load_dir, "custom_vocab.txt")):
            tokenizer.add_custom_vocab(os.path.join(load_dir, "custom_vocab.txt"))
        processor_type = config["processor"]
        return cls.load(processor_type, None, tokenizer, config["max_seq_len"])

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        create_folder(save_dir)
        config = {}
        config["tokenizer"] = self.tokenizer.__class__.__name__
        self.tokenizer.save_vocabulary(save_dir)
        # TODO make this generic to other tokenizers. We will probably want an own abstract Tokenizer
        config["lower_case"] = self.tokenizer.basic_tokenizer.do_lower_case
        config["max_seq_len"] = self.max_seq_len
        config["processor"] = self.__class__.__name__
        output_config_file = os.path.join(save_dir, "processor_config.json")
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    @abc.abstractmethod
    def _file_to_dicts(self, file: str) -> [dict]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _dict_to_samples(self, dict: dict) -> [Sample]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _sample_to_features(self, sample: Sample) -> dict:
        raise NotImplementedError()

    def _init_baskets_from_file(self, file):
        dicts = self._file_to_dicts(file)
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        self.baskets = [
            SampleBasket(raw=tr, id=f"{dataset_name}-{i}") for i, tr in enumerate(dicts)
        ]

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            basket.samples = self._dict_to_samples(basket.raw)
            for num, sample in enumerate(basket.samples):
                sample.id = f"{basket.id}-{num}"

    def _featurize_samples(self):
        for basket in self.baskets:
            for sample in basket.samples:
                sample.features = self._sample_to_features(sample=sample)

    def _create_dataset(self):
        baskets = self.baskets
        features_flat = []
        for basket in baskets:
            for sample in basket.samples:
                features_flat.extend(sample.features)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names

    def dataset_from_file(self, file):
        """
        Contains all the functionality to turn a data file into a PyTorch Dataset and a
        list of tensor names. This is used for training and evaluation.

        :param file: Name of the file containing the data.
        :type file: str
        :return: a Pytorch dataset and a list of tensor names.
        """
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
# Sequence Classification Processors ####
#########################################
class GNADProcessor(Processor):
    """
    Used to handle the GNAD dataset
    """
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename="train.csv",
        dev_filename=None,
        test_filename="test.csv",
        dev_split=0.1,
    ):

        # General Processor attributes
        label_list = [
            "Web",
            "Sport",
            "International",
            "Panorama",
            "Wissenschaft",
            "Wirtschaft",
            "Kultur",
            "Etat",
            "Inland",
        ]
        metric = "acc"
        label_dtype = torch.long

        # Custom processor attributes
        self.delimiter = ";"
        self.quote_char = "'"
        self.skiprows = [0]
        self.columns = ["label", "text"]

        super(GNADProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    def _file_to_dicts(self, file: str) -> [dict]:
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            columns=self.columns,
        )
        return dicts

    def _dict_to_samples(self, dict: dict) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(
            dict["text"], self.tokenizer, self.max_seq_len
        )
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            label_list=self.label_list,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


class GermEval18CoarseProcessor(Processor):
    """
    Used to handle the GermEval18 dataset that uses the coase labels
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
    ):

        # General Processor attributes
        self.label_list = ["OTHER", "OFFENSE"]
        self.metrics = "f1_macro"
        self.label_dtype = torch.long

        # Custom Processor attributes
        self.delimiter = "\t"
        self.skiprows = [0]
        self.columns = ["text", "label", "unused"]

        super(GermEval18CoarseProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=self.label_list,
            metrics=self.metrics,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=self.label_dtype,
        )

    def _file_to_dicts(self, file: str) -> dict:
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            columns=self.columns,
        )
        return dicts

    def _dict_to_samples(self, dict: dict) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(
            dict["text"], self.tokenizer, self.max_seq_len
        )
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            label_list=self.label_list,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


class GermEval18FineProcessor(Processor):
    """
    Used to handle the GermEval18 dataset that uses the fine labels
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
    ):

        # General Processor attributes
        label_list = ["OTHER", "INSULT", "ABUSE", "PROFANITY"]
        metric = "f1_macro"
        label_dtype = torch.long

        # Custom Processor attributes
        self.delimiter = "\t"
        self.skiprows = [0]
        # self.text_index = 0
        # self.label_index = 2
        self.columns = ["text", "unused", "label"]

        super(GermEval18FineProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    def _file_to_dicts(self, file: str) -> dict:
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            columns=self.columns,
        )
        return dicts

    def _dict_to_samples(self, dict: dict) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(
            dict["text"], self.tokenizer, self.max_seq_len
        )
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            label_list=self.label_list,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


#####################
# NER Processors ####
#####################
class CONLLProcessor(Processor):
    """
    Used to handle the CoNLL 2003 dataset (https://www.clips.uantwerpen.be/conll2003/ner/)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_file="train.txt",
        dev_file="dev.txt",
        test_file="test.txt",
        dev_split=0.0,
    ):
        # General Processor attributes
        label_list = [
            "[PAD]",
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "X",
            "B-OTH",
            "I-OTH",
        ]
        label_dtype = torch.long
        metric = "seq_f1"

        super(CONLLProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metric,
            train_filename=train_file,
            dev_filename=dev_file,
            test_filename=test_file,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    def _file_to_dicts(self, file: str) -> [dict]:
        dicts = read_ner_file(filename=file)
        return dicts

    def _dict_to_samples(self, dict: dict) -> [Sample]:
        # this tokenization also stores offsets, which helps to map our entity tags back to original positions
        tokenized = tokenize_with_metadata(
            dict["text"], self.tokenizer, self.max_seq_len
        )
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_ner(
            sample=sample,
            label_list=self.label_list,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


class GermEval14Processor(Processor):
    """
    Used to handle the GermEval14 dataset (https://www.clips.uantwerpen.be/conll2003/ner/)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_file="train.txt",
        dev_file="dev.txt",
        test_file="test.txt",
        dev_split=0.0,
    ):
        # General Processor attributes
        label_list = [
            "[PAD]",
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "X",
            "B-OTH",
            "I-OTH",
        ]
        label_dtype = torch.long
        metric = "seq_f1"
        self.delimiter = " "

        super(GermEval14Processor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metrics=metric,
            train_filename=train_file,
            dev_filename=dev_file,
            test_filename=test_file,
            dev_split=dev_split,
            data_dir=data_dir,
            label_dtype=label_dtype,
        )

    def _file_to_dicts(self, file: str) -> [dict]:
        dicts = read_ner_file(filename=file, sep=self.delimiter)
        return dicts

    def _dict_to_samples(self, dict: dict) -> [Sample]:
        # this tokenization also stores offsets, which helps to map our entity tags back to original positions
        tokenized = tokenize_with_metadata(
            dict["text"], self.tokenizer, self.max_seq_len
        )
        return [Sample(id=None, clear_text=dict, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_ner(
            sample=sample,
            label_list=self.label_list,
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
    ):
        # General Processor attributes
        label_list = [list(tokenizer.vocab), ["True", "False"]]  # labels for both heads
        metrics = ["acc", "acc"]
        label_dtype = torch.long

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
        )

    def _file_to_dicts(self, file: str) -> list:
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter)
        return dicts

    def _init_samples_in_baskets(self):
        """ Overriding the method of the parent class here, because in this case we cannot simply convert one dict to samples.
        We need to know about the other dicts as well since we want with prob 50% to use sentences of other docs!
        So we operate directly on the baskets"""
        self.baskets = create_samples_sentence_pairs(self.baskets)

    def _dict_to_samples(self, dict):
        raise NotImplementedError

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_bert_lm(
            sample=sample, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer
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
        doc_stride=128,
        max_query_length=64,
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
        """
        label_list = ["start_token", "end_token"]

        metrics = ["squad"]
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        label_dtype = torch.long  # TODO check if that is correct and needed
        self.target = "classification"
        self.ph_output_type = "per_token_squad"
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

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
            label_dtype=label_dtype,
        )
        self.data = {}
        self.counts = {}
        self.stage = None

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

    def _convert_inference(self, infer_dict):
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

    def _dict_to_samples(self, dict: dict) -> [Sample]:
        # TODO split samples that are too long in this function, related to todo in self._sample_to_features
        if "paragraphs" not in dict:  # TODO change this inference mode hack
            dict = self._convert_inference(infer_dict=dict)
        samples = create_samples_squad(entry=dict)
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
        )
        return features
