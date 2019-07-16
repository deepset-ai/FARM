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
from torch.utils.data import random_split
from farm.data_handler.samples import create_sample_ner, create_samples_sentence_pairs
from farm.data_handler.input_features import (
    samples_to_features_sequence,
    samples_to_features_ner,
    samples_to_features_bert_lm,
)
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.samples import (
    create_sample_one_label_one_text,
    Sample,
    SampleBasket,
)
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
        # TODO I would rather see the metric as a property of each prediction head (having a default value that however can be changed at init)
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        self.data_dir = data_dir
        self.label_dtype = label_dtype

        self.data = {}
        self.counts = {}
        self.stage = None

        self.log_params()

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

    def ensure_dev(self):
        assert self.stage == "dataset"
        # TODO Have some printout to say if dev is split or not
        # TODO checks to ensure dev is loaded the right way

        if "dev" not in self.data:
            n_dev = int(self.dev_split * self.counts["train"])
            n_train = self.counts["train"] - n_dev
            self.counts["train"] = n_train
            self.counts["dev"] = n_dev

            # Todo: Seed
            train_dataset, dev_dataset = random_split(
                self.data["train"], [n_train, n_dev]
            )
            self.data["train"] = train_dataset
            self.data["dev"] = dev_dataset

    @abc.abstractmethod
    def read_from_file(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_samples(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def featurize_samples(self):
        raise NotImplementedError()

    def create_dataset(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            features_flat = []
            for basket in baskets:
                for sample in basket.samples:
                    features_flat.append(sample.features)
            self.data[dataset_name], self.tensor_names = convert_features_to_dataset(
                features=features_flat
            )
        self.stage = "dataset"

    def count_samples(self):
        for dataset_name in self.data:
            count = 0
            baskets = self.data[dataset_name]
            for basket in baskets:
                count += len(basket.samples)
            self.counts[dataset_name] = count

    def dataset_from_file(self):
        self.read_from_file()
        self.create_samples()
        self.count_samples()
        self.featurize_samples()
        self.log_samples(3)
        self.create_dataset()
        self.ensure_dev()
        return self.data["train"], self.data["dev"], self.data["test"]

    def dataset_from_raw_data(self, raw_data):
        self.data["inference"] = [
            SampleBasket(raw=tr, id="infer - {}".format(i))
            for i, tr in enumerate(raw_data)
        ]
        self.create_samples()
        self.count_samples()
        self.featurize_samples()
        self.create_dataset()
        return self.data["inference"]

    def log_samples(self, n_samples):
        for dataset_name, buckets in self.data.items():
            logger.info(
                "*** Show {} random examples from {} dataset ***".format(
                    n_samples, dataset_name
                )
            )
            for i in range(n_samples):
                random_bucket = random.choice(buckets)
                random_sample = random.choice(random_bucket.samples)
                logger.info(random_sample)

    def log_params(self):
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

        # TODO Find neater way to do this
        metric = "acc"
        dev_split = dev_split
        label_dtype = torch.long

        # custom processor attributes
        self.target = "classification"
        self.delimiter = ";"

        # # TODO: Is this inheritance needed?
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

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_tsv(filename=train_file, delimiter=self.delimiter)
        self.data["train"] = [
            SampleBasket(raw=tr, id="train - {}".format(i))
            for i, tr in enumerate(train_raw)
        ]

        test_raw = read_tsv(filename=test_file, delimiter=self.delimiter)
        self.data["test"] = [
            SampleBasket(raw=tr, id="test - {}".format(i))
            for i, tr in enumerate(test_raw)
        ]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_tsv(filename=dev_file, delimiter=self.delimiter)
            self.data["dev"] = [
                SampleBasket(raw=dr, id="dev - {}".format(i))
                for i, dr in enumerate(dev_raw)
            ]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_one_label_one_text(
                    basket.raw, text_index=1, label_index=0, basket_id=basket.id
                )
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:

                features = samples_to_features_sequence(
                    samples=basket.samples,
                    label_list=self.label_list,
                    max_seq_len=self.max_seq_len,
                    tokenizer=self.tokenizer,
                    target=self.target,
                )
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"


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

        label_list = ["OTHER", "OFFENSE"]

        # TODO Find neater way to do this
        metric = "f1_macro"
        dev_split = dev_split
        label_dtype = torch.long

        self.target = "classification"
        self.delimiter = "\t"
        self.skip_first_line = True
        self.text_index = 0
        self.label_index = 1

        # # TODO: Is this inheritance needed?
        super(GermEval18CoarseProcessor, self).__init__(
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

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_tsv(
            filename=train_file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.data["train"] = [
            SampleBasket(raw=tr, id="train - {}".format(i))
            for i, tr in enumerate(train_raw)
        ]

        test_raw = read_tsv(
            filename=test_file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.data["test"] = [
            SampleBasket(raw=tr, id="test - {}".format(i))
            for i, tr in enumerate(test_raw)
        ]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_tsv(
                filename=dev_file,
                delimiter=self.delimiter,
                skip_first_line=self.skip_first_line,
            )
            self.data["dev"] = [
                SampleBasket(raw=dr, id="dev - {}".format(i))
                for i, dr in enumerate(dev_raw)
            ]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_one_label_one_text(
                    basket.raw,
                    text_index=self.text_index,
                    label_index=self.label_index,
                    basket_id=basket.id,
                )
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:

                features = samples_to_features_sequence(
                    samples=basket.samples,
                    label_list=self.label_list,
                    max_seq_len=self.max_seq_len,
                    tokenizer=self.tokenizer,
                    target=self.target,
                )
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"


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

        label_list = ["OTHER", "INSULT", "ABUSE", "PROFANITY"]

        # TODO Find neater way to do this
        metric = "f1_macro"
        dev_split = dev_split
        label_dtype = torch.long

        self.target = "classification"
        self.delimiter = "\t"
        self.skip_first_line = True
        self.text_index = 0
        self.label_index = 2

        # # TODO: Is this inheritance needed?
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

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_tsv(
            filename=train_file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.data["train"] = [
            SampleBasket(raw=tr, id="train - {}".format(i))
            for i, tr in enumerate(train_raw)
        ]

        test_raw = read_tsv(
            filename=test_file,
            delimiter=self.delimiter,
            skip_first_line=self.skip_first_line,
        )
        self.data["test"] = [
            SampleBasket(raw=tr, id="test - {}".format(i))
            for i, tr in enumerate(test_raw)
        ]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_tsv(
                filename=dev_file,
                delimiter=self.delimiter,
                skip_first_line=self.skip_first_line,
            )
            self.data["dev"] = [
                SampleBasket(raw=dr, id="dev - {}".format(i))
                for i, dr in enumerate(dev_raw)
            ]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_one_label_one_text(
                    basket.raw,
                    text_index=self.text_index,
                    label_index=self.label_index,
                    basket_id=basket.id,
                )
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:

                features = samples_to_features_sequence(
                    samples=basket.samples,
                    label_list=self.label_list,
                    max_seq_len=self.max_seq_len,
                    tokenizer=self.tokenizer,
                    target=self.target,
                )
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"


class CONLLProcessor(Processor):
    """ Used to handle the CoNLL 2003 dataset (https://www.clips.uantwerpen.be/conll2003/ner/)"""

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_file="train.txt",
        dev_file="valid.txt",
        test_file="test.txt",
        dev_split=0.0,
    ):

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

        train_filename = train_file
        dev_filename = dev_file
        test_filename = test_file
        label_dtype = torch.long
        metric = "seq_f1"

        self.target = "classification"

        super(CONLLProcessor, self).__init__(
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

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_ner_file(filename=train_file)
        self.data["train"] = [
            SampleBasket(raw=tr, id="train - {}".format(i))
            for i, tr in enumerate(train_raw)
        ]

        test_raw = read_ner_file(filename=test_file)
        self.data["test"] = [
            SampleBasket(raw=tr, id="test - {}".format(i))
            for i, tr in enumerate(test_raw)
        ]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_ner_file(filename=dev_file)
            self.data["dev"] = [
                SampleBasket(raw=dr, id="dev - {}".format(i))
                for i, dr in enumerate(dev_raw)
            ]
        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_ner(
                    split_text=basket.raw[0], label=basket.raw[1], basket_id=basket.id
                )
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                features = samples_to_features_ner(
                    samples=basket.samples,
                    label_list=self.label_list,
                    max_seq_len=self.max_seq_len,
                    tokenizer=self.tokenizer,
                    target=self.target,
                )
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"


class GermEval14Processor(Processor):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_file="train.txt",
        dev_file="valid.txt",
        test_file="test.txt",
        dev_split=0.0,
    ):

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

        train_filename = train_file
        dev_filename = dev_file
        test_filename = test_file
        label_dtype = torch.long
        metric = "seq_f1"

        self.target = "classification"

        super(GermEval14Processor, self).__init__(
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

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_ner_file(filename=train_file)
        self.data["train"] = [
            SampleBasket(raw=tr, id="train - {}".format(i))
            for i, tr in enumerate(train_raw)
        ]

        test_raw = read_ner_file(filename=test_file)
        self.data["test"] = [
            SampleBasket(raw=tr, id="test - {}".format(i))
            for i, tr in enumerate(test_raw)
        ]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_ner_file(filename=dev_file)
            self.data["dev"] = [
                SampleBasket(raw=dr, id="dev - {}".format(i))
                for i, dr in enumerate(dev_raw)
            ]
        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_ner(
                    split_text=basket.raw[0], label=basket.raw[1], basket_id=basket.id
                )
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                features = samples_to_features_ner(
                    samples=basket.samples,
                    label_list=self.label_list,
                    max_seq_len=self.max_seq_len,
                    tokenizer=self.tokenizer,
                    target=self.target,
                )
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"


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

        # TODO how best to format this
        label_list = []
        metrics = ["acc", "acc"]
        self.delimiter = ""

        dev_split = dev_split
        label_dtype = torch.long

        # # TODO: Is this inheritance needed?
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
        self.data = {}
        self.counts = {}
        self.stage = None

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_docs_from_txt(filename=train_file, delimiter=self.delimiter)
        self.data["train"] = [
            SampleBasket(raw=tr, id="train - {}".format(i))
            for i, tr in enumerate(train_raw)
        ]

        test_raw = read_docs_from_txt(filename=test_file, delimiter=self.delimiter)
        self.data["test"] = [
            SampleBasket(raw=tr, id="test - {}".format(i))
            for i, tr in enumerate(test_raw)
        ]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_docs_from_txt(filename=dev_file, delimiter=self.delimiter)
            self.data["dev"] = [
                SampleBasket(raw=dr, id="dev - {}".format(i))
                for i, dr in enumerate(dev_raw)
            ]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name, baskets in self.data.items():
            baskets = create_samples_sentence_pairs(baskets)
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name, baskets in self.data.items():
            for basket in baskets:
                features = samples_to_features_bert_lm(
                    samples=basket.samples,
                    max_seq_len=self.max_seq_len,
                    tokenizer=self.tokenizer,
                )
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"
