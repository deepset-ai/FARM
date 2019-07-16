import logging

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from farm.data_handler.dataloader import NamedDataLoader
from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


class DataSilo(object):
    """ Loads the train, dev and test sets from file, calculates statistics from the data, casts datasets to
    PyTorch DataLoader objects. """

    def __init__(self, processor, batch_size, distributed=False):
        self.distributed = distributed
        self.processor = processor
        self.batch_size = batch_size
        self.class_weights = None

        self.load_data()

    def load_data(self):
        # fmt: off

        train_file = os.path.join(self.processor.data_dir, self.processor.train_filename)
        test_file = os.path.join(self.processor.data_dir, self.processor.test_filename)

        logger.info("Loading train set from: {}".format(train_file))
        if not self.processor.dev_filename:
            logger.info("Loading dev set as a slice of train set")
        else:
            dev_file = os.path.join(self.processor.data_dir, self.processor.dev_filename)
            logger.info("Loading dev set from: {}".format(dev_file))
        logger.info("Loading test set from: {}".format(test_file))

        dataset_train, dataset_dev, dataset_test = self.processor.dataset_from_file()

        self.calculate_statistics(dataset_train, dataset_dev, dataset_test)
        self.calculate_class_weights(dataset_train)
        self.initialize_data_loaders(dataset_train, dataset_dev, dataset_test)
        # fmt: on

    def initialize_data_loaders(self, dataset_train, dataset_dev, dataset_test):
        if self.distributed:
            sampler_train = DistributedSampler(dataset_train)
        else:
            sampler_train = RandomSampler(dataset_train)

        data_loader_train = NamedDataLoader(
            dataset=dataset_train,
            sampler=sampler_train,
            batch_size=self.batch_size,
            tensor_names=self.processor.tensor_names,
        )

        data_loader_dev = NamedDataLoader(
            dataset=dataset_dev,
            sampler=SequentialSampler(dataset_dev),
            batch_size=self.batch_size,
            tensor_names=self.processor.tensor_names,
        )

        data_loader_test = NamedDataLoader(
            dataset=dataset_test,
            sampler=SequentialSampler(dataset_test),
            batch_size=self.batch_size,
            tensor_names=self.processor.tensor_names,
        )

        self.loaders = {
            "train": data_loader_train,
            "dev": data_loader_dev,
            "test": data_loader_test,
        }

    def calculate_statistics(self, dataset_train, dataset_dev, dataset_test):
        self.counts = {
            "train": len(dataset_train),
            "dev": len(dataset_dev),
            "test": len(dataset_test),
        }

        logger.info("Examples in train: {}".format(len(dataset_train)))
        logger.info("Examples in dev  : {}".format(len(dataset_dev)))
        logger.info("Examples in test : {}".format(len(dataset_test)))

        MlLogger.log_params(
            {
                "n_samples_train": len(dataset_train),
                "n_samples_dev": len(dataset_dev),
                "n_samples_test": len(dataset_test),
            }
        )

    # TODO: maybe this can be inside calculate_statistics
    def calculate_class_weights(self, dataset):
        try:
            labels = [x[3].item() for x in dataset]
            self.class_weights = list(
                compute_class_weight("balanced", np.unique(labels), labels)
            )
        except ValueError:
            logger.info(
                "Class weighting not available for token level tasks such as NER"
            )

    def get_data_loader(self, dataset):
        return self.loaders[dataset]

    def n_samples(self, dataset):
        return self.counts[dataset]
