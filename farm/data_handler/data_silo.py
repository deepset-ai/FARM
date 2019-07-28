import logging

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from farm.data_handler.dataloader import NamedDataLoader
from farm.utils import MLFlowLogger as MlLogger
from farm.data_handler.processor import Processor

logger = logging.getLogger(__name__)


class DataSilo(object):
    """ Generates and stores PyTorch DataLoader objects for the train, dev and test datasets.
    Relies upon functionality in the processor to do the conversion of the data. Will also
    calculate and display some statistics.
     """

    def __init__(self, processor, batch_size, distributed=False):
        """
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :type processor: Processor
        :param batch_size: The size of batch that should be returned by the DataLoaders.
        :type batch_size: int
        :param distributed: Set to True if the program is running in a distributed setting.
        :type distributed: bool

        """
        self.distributed = distributed
        self.processor = processor
        self.data = {}
        self.batch_size = batch_size
        self.class_weights = None
        self._load_data()

    def _load_data(self):
        # train data
        train_file = os.path.join(self.processor.data_dir, self.processor.train_filename)
        logger.info("Loading train set from: {}".format(train_file))
        self.data["train"], self.tensor_names = self.processor.dataset_from_file(train_file)


        # dev data
        if not self.processor.dev_filename:
            logger.info("Loading dev set as a slice of train set")
            self._create_dev_from_train()
        else:
            dev_file = os.path.join(self.processor.data_dir, self.processor.dev_filename)
            logger.info("Loading dev set from: {}".format(dev_file))
            self.data["dev"], _ = self.processor.dataset_from_file(dev_file)

        # test data
        if self.processor.test_filename:
            test_file = os.path.join(self.processor.data_dir, self.processor.test_filename)
            logger.info("Loading test set from: {}".format(test_file))
            self.data["test"], _ = self.processor.dataset_from_file(test_file)

        # derive stats and meta data
        self._calculate_statistics()
        self._calculate_class_weights(self.data["train"])
        self._initialize_data_loaders()
        # fmt: on

    def _initialize_data_loaders(self):
        if self.distributed:
            sampler_train = DistributedSampler(self.data["train"])
        else:
            sampler_train = RandomSampler(self.data["train"])

        data_loader_train = NamedDataLoader(
            dataset=self.data["train"],
            sampler=sampler_train,
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )

        data_loader_dev = NamedDataLoader(
            dataset=self.data["dev"],
            sampler=SequentialSampler(self.data["dev"]),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )

        if self.processor.test_filename:
            data_loader_test = NamedDataLoader(
                dataset=self.data["test"],
                sampler=SequentialSampler(self.data["test"]),
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_test = None

        self.loaders = {
            "train": data_loader_train,
            "dev": data_loader_dev,
            "test": data_loader_test,
        }

    def _create_dev_from_train(self):
        # TODO checks to ensure dev is loaded the right way
        n_dev = int(self.processor.dev_split * len(self.data["train"]))
        n_train = len(self.data["train"]) - n_dev

        # Todo: Seed
        train_dataset, dev_dataset = random_split(self.data["train"], [n_train, n_dev])
        self.data["train"] = train_dataset
        self.data["dev"] = dev_dataset

        logger.info(
            f"Took {n_dev} samples out of train set to create dev set (dev split = {self.processor.dev_split})"
        )

    def _calculate_statistics(self,):
        self.counts = {
            "train": len(self.data["train"]),
            "dev": len(self.data["dev"]),
            "test": len(self.data.get("test", [])),
        }

        train_input_numpy = self.data["train"][:][0].numpy()
        seq_lens = np.sum(train_input_numpy != 0, axis=1)
        self.ave_len = np.mean(seq_lens)
        max_seq_len = self.data["train"][:][0].shape[1]
        self.clipped = np.mean(seq_lens == max_seq_len)


        logger.info("Examples in train: {}".format(self.counts["train"]))
        logger.info("Examples in dev  : {}".format(self.counts["dev"]))
        logger.info("Examples in test : {}".format(self.counts["test"]))
        logger.info("")
        logger.info("Max sequence length:     {}".format(max(seq_lens)))
        logger.info("Average sequence length: {}".format(self.ave_len))
        logger.info("Proportion clipped:      {}".format(self.clipped))
        if self.clipped > 0.5:
            logger.info("[Farmer's Tip] {}% of your samples got cut down to {} tokens. "
                        "Consider increasing max_seq_len. "
                        "This will lead to higher memory consumption but is likely to "
                        "improve your model performance".format(round(self.clipped * 100, 1), max_seq_len))

        MlLogger.log_params(
            {
                "n_samples_train": self.counts["train"],
                "n_samples_dev": self.counts["dev"],
                "n_samples_test": self.counts["test"],
                "ave_seq_len": self.ave_len,
                "clipped": self.clipped
            }
        )

    # TODO: maybe this can be inside calculate_statistics
    # TODO: this also computes weights for QA. What is inside x[3].item() o_O ???
    def _calculate_class_weights(self, dataset):
        try:
            labels = [x[3].item() for x in dataset]
            self.class_weights = list(
                compute_class_weight("balanced", np.unique(labels), labels)
            )
            logger.info(f"Using class weights: {self.class_weights}")
        except ValueError:
            logger.info(
                "Class weighting is currently only available for sequence classification tasks "
            )

    def _get_data_loader(self, dataset):
        return self.loaders[dataset]

    def n_samples(self, dataset):
        """
        Returns the number of samples in a given dataset.

        :param dataset: Choose from train, dev or test
        """
        return self.counts[dataset]
