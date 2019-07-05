import torch
import os
import abc
from abc import ABC
import random
import logging

from farm.data_handler.utils import read_tsv, read_docs_from_txt,read_ner_file
from torch.utils.data import random_split
from farm.data_handler.samples import (
    create_samples_gnad,
    create_samples_conll_03,
    create_sample_ner,
    create_examples_germ_eval_18_coarse,
    create_examples_germ_eval_18_fine,
    create_samples_sentence_pairs,
)
from farm.data_handler.input_features import (
    samples_to_features_sequence,
    samples_to_features_ner,
    samples_to_features_bert_lm,
)
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.samples import create_sample_one_label_one_text, Sample, SampleBasket


logger = logging.getLogger(__name__)


class Processor(ABC):
    # TODO think about how to define this parent class so it enforces that certain attributes are initialized
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 label_list,
                 metric,
                 train_filename,
                 dev_filename,
                 test_filename,
                 dev_split,
                 data_dir,
                 ph_output_type,
                 label_dtype=torch.long,
                 ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_list = label_list
        self.metric = metric
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        self.data_dir = data_dir
        self.ph_output_type = ph_output_type
        self.label_dtype = label_dtype

        self.data = {}
        self.counts = {}
        self.stage = None

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

    def dataset_from_list(self, list):
        #TODO we need to pass list to create_samples properly
        self.create_samples()
        self.count_samples()
        self.featurize_samples()
        self.create_dataset()

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
                print(random_sample)


class GNADProcessor(Processor):
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
                 train_filename="train.csv",
                 dev_filename=None,
                 test_filename="test.csv",
                 dev_split=0.1):

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
        ph_output_type = "per_sequence"

        # custom processor attributes
        self.target = "classification"
        self.delimiter = ";"

        # # TODO: Is this inheritance needed?
        super(GNADProcessor, self).__init__(tokenizer=tokenizer,
                                            max_seq_len=max_seq_len,
                                            label_list=label_list,
                                            metric=metric,
                                            train_filename=train_filename,
                                            dev_filename=dev_filename,
                                            test_filename=test_filename,
                                            dev_split=dev_split,
                                            data_dir=data_dir,
                                            ph_output_type=ph_output_type,
                                            label_dtype=label_dtype,)


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
                basket.samples = create_sample_one_label_one_text(basket.raw,
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



class GermEval18CoarseProcessor(Processor):
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
                 train_filename="train.tsv",
                 dev_filename=None,
                 test_filename="test.tsv",
                 dev_split=0.1):

        label_list = ["OTHER", "OFFENSE"]

        # TODO Find neater way to do this
        metric = "f1_macro"
        dev_split = dev_split
        label_dtype = torch.long
        ph_output_type = "per_sequence"

        self.target = "classification"
        self.delimiter = "\t"
        self.skip_first_line = True
        self.text_index = 0
        self.label_index = 1

        # # TODO: Is this inheritance needed?
        super(GermEval18CoarseProcessor, self).__init__(tokenizer=tokenizer,
                                                        max_seq_len=max_seq_len,
                                                        label_list=label_list,
                                                        metric=metric,
                                                        train_filename=train_filename,
                                                        dev_filename=dev_filename,
                                                        test_filename=test_filename,
                                                        dev_split=dev_split,
                                                        data_dir=data_dir,
                                                        ph_output_type=ph_output_type,
                                                        label_dtype=label_dtype,)

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_tsv(filename=train_file,
                              delimiter=self.delimiter,
                             skip_first_line=self.skip_first_line)
        self.data["train"] = [SampleBasket(raw=tr, id="train - {}".format(i)) for i, tr in enumerate(train_raw)]

        test_raw = read_tsv(filename=test_file,
                            delimiter=self.delimiter,
                            skip_first_line=self.skip_first_line)
        self.data["test"] = [SampleBasket(raw=tr, id="test - {}".format(i)) for i, tr in enumerate(test_raw)]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_tsv(filename=dev_file,
                               delimiter=self.delimiter,
                               skip_first_line=self.skip_first_line)
            self.data["dev"] = [SampleBasket(raw=dr, id="dev - {}".format(i)) for i, dr in enumerate(dev_raw)]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_one_label_one_text(basket.raw,
                                                                    text_index=self.text_index,
                                                                    label_index=self.label_index,
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



class GermEval18FineProcessor(Processor):
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
                 train_filename="train.tsv",
                 dev_filename=None,
                 test_filename="test.tsv",
                 dev_split=0.1):

        label_list = ["OTHER", "INSULT", "ABUSE", "PROFANITY"]

        # TODO Find neater way to do this
        metric = "f1_macro"
        dev_split = dev_split
        label_dtype = torch.long
        ph_output_type = "per_sequence"

        self.target = "classification"
        self.delimiter = "\t"
        self.skip_first_line = True
        self.text_index = 0
        self.label_index = 2

        # # TODO: Is this inheritance needed?
        super(GermEval18FineProcessor, self).__init__(tokenizer=tokenizer,
                                                        max_seq_len=max_seq_len,
                                                        label_list=label_list,
                                                        metric=metric,
                                                        train_filename=train_filename,
                                                        dev_filename=dev_filename,
                                                        test_filename=test_filename,
                                                        dev_split=dev_split,
                                                        data_dir=data_dir,
                                                        ph_output_type=ph_output_type,
                                                        label_dtype=label_dtype,)

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_tsv(filename=train_file,
                              delimiter=self.delimiter,
                             skip_first_line=self.skip_first_line)
        self.data["train"] = [SampleBasket(raw=tr, id="train - {}".format(i)) for i, tr in enumerate(train_raw)]

        test_raw = read_tsv(filename=test_file,
                            delimiter=self.delimiter,
                            skip_first_line=self.skip_first_line)
        self.data["test"] = [SampleBasket(raw=tr, id="test - {}".format(i)) for i, tr in enumerate(test_raw)]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_tsv(filename=dev_file,
                               delimiter=self.delimiter,
                               skip_first_line=self.skip_first_line)
            self.data["dev"] = [SampleBasket(raw=dr, id="dev - {}".format(i)) for i, dr in enumerate(dev_raw)]

        self.stage = "lines"

    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_one_label_one_text(basket.raw,
                                                                    text_index=self.text_index,
                                                                    label_index=self.label_index,
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


class CONLLProcessor(Processor):
    """ Used to handle the CoNLL 2003 dataset (https://www.clips.uantwerpen.be/conll2003/ner/)"""

    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
                 train_file="train.txt",
                 dev_file="valid.txt",
                 test_file="test.txt",
                 dev_split=0.0):

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
        ph_output_type = "per_token"

        self.target = "classification"

        super(CONLLProcessor, self).__init__(tokenizer=tokenizer,
                                             max_seq_len=max_seq_len,
                                             label_list=label_list,
                                             metric=metric,
                                             train_filename=train_filename,
                                             dev_filename=dev_filename,
                                             test_filename=test_filename,
                                             dev_split=dev_split,
                                             data_dir=data_dir,
                                             ph_output_type=ph_output_type,
                                             label_dtype=label_dtype)

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_ner_file(filename=train_file)
        self.data["train"] = [SampleBasket(raw=tr, id="train - {}".format(i)) for i, tr in enumerate(train_raw)]

        test_raw = read_ner_file(filename=test_file)
        self.data["test"] = [SampleBasket(raw=tr, id="test - {}".format(i)) for i, tr in enumerate(test_raw)]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_ner_file(filename=dev_file)
            self.data["dev"] = [SampleBasket(raw=dr, id="dev - {}".format(i)) for i, dr in enumerate(dev_raw)]
        self.stage = "lines"


    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_ner(split_text=basket.raw[0],
                                                   label=basket.raw[1],
                                                   basket_id=basket.id)
        self.stage = "examples"


    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                features = samples_to_features_ner(samples=basket.samples,
                                                    label_list=self.label_list,
                                                    max_seq_len=self.max_seq_len,
                                                    tokenizer=self.tokenizer,
                                                    target=self.target)
                for sample, feat in zip(basket.samples, features):
                    sample.features = feat
        self.stage = "features"



class GermEval14Processor(Processor):

    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 data_dir,
                 train_file="train.txt",
                 dev_file="valid.txt",
                 test_file="test.txt",
                 dev_split=0.0):

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
        ph_output_type = "per_token"

        self.target = "classification"

        super(GermEval14Processor, self).__init__(tokenizer=tokenizer,
                                                     max_seq_len=max_seq_len,
                                                     label_list=label_list,
                                                     metric=metric,
                                                     train_filename=train_filename,
                                                     dev_filename=dev_filename,
                                                     test_filename=test_filename,
                                                     dev_split=dev_split,
                                                     data_dir=data_dir,
                                                     ph_output_type=ph_output_type,
                                                     label_dtype=label_dtype)

    def read_from_file(self):
        train_file = os.path.join(self.data_dir, self.train_filename)
        test_file = os.path.join(self.data_dir, self.test_filename)

        train_raw = read_ner_file(filename=train_file)
        self.data["train"] = [SampleBasket(raw=tr, id="train - {}".format(i)) for i, tr in enumerate(train_raw)]

        test_raw = read_ner_file(filename=test_file)
        self.data["test"] = [SampleBasket(raw=tr, id="test - {}".format(i)) for i, tr in enumerate(test_raw)]

        if self.dev_filename:
            dev_file = os.path.join(self.data_dir, self.dev_filename)
            dev_raw = read_ner_file(filename=dev_file)
            self.data["dev"] = [SampleBasket(raw=dr, id="dev - {}".format(i)) for i, dr in enumerate(dev_raw)]
        self.stage = "lines"


    def create_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                basket.samples = create_sample_ner(split_text=basket.raw[0],
                                                   label=basket.raw[1],
                                                   basket_id=basket.id)
        self.stage = "examples"


    def featurize_samples(self):
        for dataset_name in self.data:
            baskets = self.data[dataset_name]
            for basket in baskets:
                features = samples_to_features_ner(samples=basket.samples,
                                                    label_list=self.label_list,
                                                    max_seq_len=self.max_seq_len,
                                                    tokenizer=self.tokenizer,
                                                    target=self.target)
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
        metric = "acc"
        self.delimiter = ""

        dev_split = dev_split
        label_dtype = torch.long
        # TODO adjust this to new cases
        self.ph_output_type = "per_sequence"

        # # TODO: Is this inheritance needed?
        super(BertStyleLMProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metric=metric,
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
