import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class DataBunch(object):
    """ Loads the train, dev and test sets from file, calculates statistics from the data, casts datasets to
    PyTorch DataLoader objects. """

    def __init__(self, preprocessing_pipeline, batch_size, distributed=False):
        self.distributed = distributed
        self.pipeline = preprocessing_pipeline
        self.batch_size = batch_size
        self.class_weights = None

        self.load_data()

    def load_data(self):
        # fmt: off
        train_file = self.pipeline.train_file
        dev_file = self.pipeline.dev_file
        test_file = self.pipeline.test_file

        logger.info("Loading train set from: {}".format(train_file))
        if not dev_file:
            logger.info("Loading dev set as a slice of train set")
        else:
            logger.info("Loading dev set from: {}".format(dev_file))
        logger.info("Loading test set from: {}".format(test_file))

        # TODO checks to ensure dev is loaded the right way
        if not self.pipeline.dev_split:
            dataset_train = self.pipeline.convert(train_file, start="file", end="dataset")
            dataset_dev = self.pipeline.convert(dev_file, start="file", end="dataset")

        else:
            # TODO How to handle seed?
            SEED = 42
            list_train_dev = self.pipeline.convert(train_file, start="file", end="list")
            list_train, list_dev = train_test_split(list_train_dev,
                                                    test_size=self.pipeline.dev_split,
                                                    random_state=SEED)

            dataset_train = self.pipeline.convert(list_train, start="list", end="dataset")
            dataset_dev = self.pipeline.convert(list_dev, start="list", end="dataset")

        dataset_test = self.pipeline.convert(test_file, start="file", end="dataset")

        self.calculate_statistics(dataset_train, dataset_dev, dataset_test)
        self.calculate_class_weights(dataset_train)
        self.initialize_data_loaders(dataset_train, dataset_dev, dataset_test)
        # fmt: on

    def initialize_data_loaders(self, dataset_train, dataset_dev, dataset_test):
        if self.distributed:
            sampler_train = DistributedSampler(dataset_train)
        else:
            sampler_train = RandomSampler(dataset_train)

        data_loader_train = DataLoader(
            dataset=dataset_train, sampler=sampler_train, batch_size=self.batch_size
        )

        data_loader_dev = DataLoader(
            dataset=dataset_dev,
            sampler=SequentialSampler(dataset_dev),
            batch_size=self.batch_size,
        )

        data_loader_test = DataLoader(
            dataset=dataset_test,
            sampler=SequentialSampler(dataset_test),
            batch_size=self.batch_size,
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
