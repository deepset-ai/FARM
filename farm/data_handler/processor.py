import torch
import os
import abc
from abc import ABC
import random
import logging
import json

from pytorch_pretrained_bert.tokenization import BertTokenizer

from farm.data_handler.utils import read_tsv, read_docs_from_txt, read_ner_file
from farm.file_utils import create_folder
from farm.data_handler.samples import create_sample_ner, create_samples_sentence_pairs
from farm.data_handler.input_features import (
    samples_to_features_sequence,
    samples_to_features_ner,
    samples_to_features_bert_lm,
)
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.samples import create_sample_one_label_one_text, SampleBasket
from farm.utils import MLFlowLogger as MlLogger


logger = logging.getLogger(__name__)

TOKENIZER_MAP = {"BertTokenizer": BertTokenizer}


class Processor(ABC):
    # TODO think about how to define this parent class so it enforces that certain attributes are initialized
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
        :param processor_name:
        :param data_dir:
        :param tokenizer:
        :param max_seq_len:
        :return:
        """
        return cls.subclasses[processor_name](
            data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    @classmethod
    def load_from_dir(cls, load_dir):
        """
        Load infers the specific type of Processor from a config file (e.g. GNADProcessor) and loads an instance of it.
        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: Processor, Instance of a Processor Subclass (e.g. GNADProcessor)
        """
        # read config
        processor_config_file = os.path.join(load_dir, "processor_config.json")
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = TOKENIZER_MAP[config["tokenizer"]].from_pretrained(
            load_dir, do_lower_case=config["lower_case"]
        )
        processor_type = config["processor"]
        return cls.load(processor_type, None, tokenizer, config["max_seq_len"])

    def save(self, save_dir):
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
    def _init_baskets_from_file(self, file):
        raise NotImplementedError()

    @abc.abstractmethod
    def _init_samples_in_baskets(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _featurize_samples(self):
        raise NotImplementedError()

    def _create_dataset(self):
        baskets = self.baskets
        features_flat = []
        for basket in baskets:
            for sample in basket.samples:
                features_flat.append(sample.features)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        self.stage = "dataset"
        return dataset, tensor_names

    def dataset_from_file(self, file):
        self._init_baskets_from_file(file)
        self._init_samples_in_baskets()
        self._featurize_samples()
        self._log_samples(3)
        dataset, tensor_names = self._create_dataset()
        return dataset, tensor_names

    def dataset_from_raw_data(self, raw_data):
        self.baskets = [
            SampleBasket(raw=tr, id="infer - {}".format(i))
            for i, tr in enumerate(raw_data)
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
        MlLogger.log_params(params)


class GNADProcessor(Processor):
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
        self.skip_first_line = False
        self.text_index = 1
        self.label_index = 0

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

    def _init_baskets_from_file(self, file):
        pure_name = os.path.splitext(os.path.basename(file))[0]
        raw_data = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.baskets = [
            SampleBasket(raw=tr, id=f"{pure_name}-{i}") for i, tr in enumerate(raw_data)
        ]

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            basket.samples = create_sample_one_label_one_text(
                basket.raw,
                text_index=self.text_index,
                label_index=self.label_index,
                basket_id=basket.id,
            )

    def _featurize_samples(self):
        for basket in self.baskets:
            features = samples_to_features_sequence(
                samples=basket.samples,
                label_list=self.label_list,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
            )
            for sample, feat in zip(basket.samples, features):
                sample.features = feat


class GermEval18CoarseProcessor(Processor):
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
        label_list = ["OTHER", "OFFENSE"]
        metrics = "f1_macro"
        label_dtype = torch.long

        # Custom Processor attributes
        self.delimiter = "\t"
        self.skip_first_line = True
        self.text_index = 0
        self.label_index = 1

        super(GermEval18CoarseProcessor, self).__init__(
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

    def _init_baskets_from_file(self, file):
        pure_name = os.path.splitext(os.path.basename(file))[0]
        raw_data = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.baskets = [
            SampleBasket(raw=tr, id=f"{pure_name}-{i}") for i, tr in enumerate(raw_data)
        ]

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            basket.samples = create_sample_one_label_one_text(
                basket.raw,
                text_index=self.text_index,
                label_index=self.label_index,
                basket_id=basket.id,
            )

    def _featurize_samples(self):
        for basket in self.baskets:
            features = samples_to_features_sequence(
                samples=basket.samples,
                label_list=self.label_list,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
            )
            for sample, feat in zip(basket.samples, features):
                sample.features = feat


class GermEval18FineProcessor(Processor):
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
        self.skip_first_line = True
        self.text_index = 0
        self.label_index = 2

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

    def _init_baskets_from_file(self, file):
        pure_name = os.path.splitext(os.path.basename(file))[0]
        raw_data = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.baskets = [
            SampleBasket(raw=tr, id=f"{pure_name}-{i}") for i, tr in enumerate(raw_data)
        ]

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            basket.samples = create_sample_one_label_one_text(
                basket.raw,
                text_index=self.text_index,
                label_index=self.label_index,
                basket_id=basket.id,
            )

    def _featurize_samples(self):
        for basket in self.baskets:
            features = samples_to_features_sequence(
                samples=basket.samples,
                label_list=self.label_list,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
            )
            for sample, feat in zip(basket.samples, features):
                sample.features = feat


class CONLLProcessor(Processor):
    """ Used to handle the CoNLL 2003 dataset (https://www.clips.uantwerpen.be/conll2003/ner/)"""

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
            "[CLS]",
            "[SEP]",
        ]
        label_dtype = torch.long
        metric = "seq_f1"

        # Custom attributes
        self.split_text_index = 0
        self.label_index = 1

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

    def _init_baskets_from_file(self, file):
        pure_name = os.path.splitext(os.path.basename(file))[0]
        raw_data = read_ner_file(filename=file)
        self.baskets = [
            SampleBasket(raw=tr, id=f"{pure_name}-{i}") for i, tr in enumerate(raw_data)
        ]

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            basket.samples = create_sample_ner(
                split_text=basket.raw[self.split_text_index],
                label=basket.raw[self.label_index],
                basket_id=basket.id,
            )

    def _featurize_samples(self):
        for basket in self.baskets:
            features = samples_to_features_ner(
                samples=basket.samples,
                label_list=self.label_list,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
            )
            for sample, feat in zip(basket.samples, features):
                sample.features = feat


class GermEval14Processor(Processor):
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
            "[CLS]",
            "[SEP]",
        ]

        label_dtype = torch.long
        metric = "seq_f1"

        # Custom attributes
        self.split_text_index = 0
        self.label_index = 1

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

    def _init_baskets_from_file(self, file):
        pure_name = os.path.splitext(os.path.basename(file))[0]
        raw_data = read_ner_file(filename=file)
        self.baskets = [
            SampleBasket(raw=tr, id=f"{pure_name}-{i}") for i, tr in enumerate(raw_data)
        ]

    def _init_samples_in_baskets(self):
        for basket in self.baskets:
            basket.samples = create_sample_ner(
                split_text=basket.raw[self.split_text_index],
                label=basket.raw[self.label_index],
                basket_id=basket.id,
            )

    def _featurize_samples(self):
        for basket in self.baskets:
            features = samples_to_features_ner(
                samples=basket.samples,
                label_list=self.label_list,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
            )
            for sample, feat in zip(basket.samples, features):
                sample.features = feat


class BertStyleLMProcessor(Processor):
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
        label_list = []
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

    def _init_baskets_from_file(self, file):
        pure_name = os.path.splitext(os.path.basename(file))[0]
        raw_data = read_docs_from_txt(filename=file, delimiter=self.delimiter)
        self.baskets = [
            SampleBasket(raw=tr, id=f"{pure_name}-{i}") for i, tr in enumerate(raw_data)
        ]

    def _init_samples_in_baskets(self):
        self.baskets = create_samples_sentence_pairs(self.baskets)

    def _featurize_samples(self):
        for basket in self.baskets:
            features = samples_to_features_bert_lm(
                samples=basket.samples,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
            )
            for sample, feat in zip(basket.samples, features):
                sample.features = feat
