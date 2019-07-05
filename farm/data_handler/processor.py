import torch
import os
import abc
from abc import ABC
from farm.data_handler.utils import read_tsv
from torch.utils.data import random_split
from farm.data_handler.samples import (
    create_samples_gnad,
    create_examples_conll_03,
    create_examples_germ_eval_18_coarse,
    create_examples_germ_eval_18_fine,
    create_examples_lm,
)
from farm.data_handler.input_features import (
    samples_to_features_sequence,
    examples_to_features_ner,
    examples_to_features_lm,
)
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.samples import create_samples_one_label_one_text, Sample, SampleBasket


class Processor(ABC):
    # TODO think about how to define this parent class so it enforces that certain attributes are initialized
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 label_list,
                 metric,
                 filenames,
                 dev_split,
                 data_dir,
                 delimiter,
                 label_dtype=torch.long,
                 ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_list = label_list
        self.metric = metric
        self.filenames = filenames
        self.dev_split = dev_split
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.label_dtype = label_dtype

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
            train_dataset, dev_dataset = random_split(self.data["train"], [n_train, n_dev])
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
            self.data[dataset_name], self.tensor_names = convert_features_to_dataset(features=features_flat)
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
        self.create_dataset()
        self.ensure_dev()
        return self.data["train"], self.data["dev"], self.data["test"]

    def dataaset_from_list(self, list):
        self.create_samples()
        self.count_samples()
        self.featurize_samples()
        self.create_dataset()


class GNADProcessor(Processor):
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
                 train_filename = "train.csv",
                 dev_filename=None,
                 test_filename = "test.csv",
                 dev_split = 0.1):

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
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        filenames = [self.train_filename, self.dev_filename, self.test_filename]
        dev_split = dev_split
        label_dtype = torch.long
        delimiter = ";"
        self.target = "classification"
        self.ph_output_type = "per_sequence"

        # # TODO: Is this inheritance needed?
        super(GNADProcessor, self).__init__(tokenizer=tokenizer,
                                            max_seq_len=max_seq_len,
                                            label_list=label_list,
                                            metric=metric,
                                            filenames=filenames,
                                            dev_split=dev_split,
                                            data_dir=data_dir,
                                            delimiter=delimiter,
                                            label_dtype=label_dtype,)
        self.data = {}
        self.counts = {}
        self.stage = None

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_tsv(filename=train_file,
                              delimiter=self.delimiter)
        self.data["train"] = [SampleBasket(raw=tr, id="train - {}".format(i)) for i, tr in enumerate(train_raw)]

        test_raw = read_tsv(filename=test_file,
                              delimiter=self.delimiter)
        self.data["test"] = [SampleBasket(raw=tr, id="test - {}".format(i)) for i, tr in enumerate(test_raw)]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_tsv(filename=dev_file,
                                delimiter=self.delimiter)
            self.data["dev"] = [SampleBasket(raw=dr, id="dev - {}".format(i)) for i, dr in enumerate(dev_raw)]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_samples_one_label_one_text(basket.raw,
                                                                    text_index=1,
                                                                    label_index=0,
                                                                    basket_id=basket.id)
        self.stage = "examples"

    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:

                features = samples_to_features_sequence(samples=basket.samples,
                                                        label_list=self.label_list,
                                                        max_seq_len=self.max_seq_len,
                                                        tokenizer=self.tokenizer,
                                                        target=self.target)
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"



