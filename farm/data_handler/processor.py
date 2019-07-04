import torch
import os
from farm.data_handler.utils import read_tsv
from torch.utils.data import random_split
from farm.data_handler.input_example import (
    create_examples_gnad,
    create_examples_conll_03,
    create_examples_germ_eval_18_coarse,
    create_examples_germ_eval_18_fine,
    create_examples_lm,
)
from farm.data_handler.input_features import (
    examples_to_features_sequence,
    examples_to_features_ner,
    examples_to_features_lm,
)
from farm.data_handler.dataset import convert_features_to_dataset


class Processor:
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

    def create_examples(self):
        raise NotImplementedError()


class GNADProcessor(Processor):
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
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

        metric = "acc"
        self.train_filename = "train.csv"
        self.dev_filename = None
        self.test_filename = "test.csv"
        filenames = [self.train_filename, self.dev_filename, self.test_filename]
        dev_split = 0.1
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
                                            label_dtype=label_dtype)

        self.data = {}
        self.counts = {}
        self.stage = None


    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        self.data["train"] = read_tsv(filename=train_file,
                                      delimiter=self.delimiter)
        self.data["test"] = read_tsv(filename=test_file,
                                      delimiter=self.delimiter)

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            self.data["dev"] = read_tsv(filename=dev_file,
                                        delimiter=self.delimiter)

        self.stage = "lines"

    def create_examples(self):
        for dataset_name in self.data:
            self.data[dataset_name] = create_examples_gnad(lines=self.data[dataset_name],
                                                               set_type=dataset_name)
        self.stage = "examples"

    def create_features(self):
        for dataset_name in self.data:
            self.data[dataset_name] = examples_to_features_sequence(examples=self.data[dataset_name],
                                                                      label_list=self.label_list,
                                                                      max_seq_len=self.max_seq_len,
                                                                      tokenizer=self.tokenizer,
                                                                      target=self.target)
        self.stage = "features"

    def create_dataset(self):
        for dataset_name in self.data:
            self.data[dataset_name] = convert_features_to_dataset(features=self.data[dataset_name],
                                                                      label_dtype=self.label_dtype)
        self.stage = "dataset"

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


    def count_data(self):
        for dataset_name in self.data:
            self.counts[dataset_name] = len(self.data[dataset_name])


    def dataset_from_file(self):
        self.read_from_file()
        self.create_examples()
        self.count_data()
        self.create_features()
        self.create_dataset()
        self.ensure_dev()
        return self.data["train"], self.data["dev"], self.data["test"]

    def dataaset_from_list(self, list):
        self.create_examples()
        self.count_data()
        self.create_features()
        self.create_dataset()
