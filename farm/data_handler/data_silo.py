import copy
import logging
import torch.multiprocessing as mp
import os
from contextlib import ExitStack
from functools import partial
import random

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm

from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.processor import Processor
from farm.data_handler.utils import grouper
from farm.utils import MLFlowLogger as MlLogger
from farm.utils import log_ascii_workers
from farm.visual.ascii.images import TRACTOR_SMALL

logger = logging.getLogger(__name__)


class DataSilo:
    """ Generates and stores PyTorch DataLoader objects for the train, dev and test datasets.
    Relies upon functionality in the processor to do the conversion of the data. Will also
    calculate and display some statistics.
     """

    def __init__(self, processor, batch_size, distributed=False, multiprocessing_chunk_size=100):
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
        self.multiprocessing_chunk_size = multiprocessing_chunk_size
        self.max_processes = 128
        self._load_data()

    @classmethod
    def _multiproc(cls, chunk, processor):
        dicts = [d[1] for d in chunk]
        index = chunk[0][0]
        dataset = processor.dataset_from_dicts(dicts=dicts,index=index)
        return dataset

    def _get_dataset(self, filename):
        dicts = self.processor.file_to_dicts(filename)
        #shuffle list of dicts here if we later want to have a random dev set splitted from train set
        if self.processor.train_filename in filename:
            if not self.processor.dev_filename:
                if self.processor.dev_split > 0.0:
                    random.shuffle(dicts)

        dict_batches_to_process = int(len(dicts) / self.multiprocessing_chunk_size)
        num_cpus = min(mp.cpu_count(), self.max_processes,  dict_batches_to_process) or 1

        with ExitStack() as stack:
            p = stack.enter_context(mp.Pool(processes=num_cpus))

            logger.info(
                f"Got ya {num_cpus} parallel workers to convert dict chunks to datasets (chunksize = {self.multiprocessing_chunk_size})..."
            )
            log_ascii_workers(num_cpus, logger)

            results = p.imap(
                partial(self._multiproc, processor=self.processor),
                grouper(dicts, self.multiprocessing_chunk_size),
                chunksize=1,
            )

            datasets = []
            with tqdm(total=len(dicts), unit=' Dicts') as pbar:
                for dataset, tensor_names in results:
                    datasets.append(dataset)
                    pbar.update(self.multiprocessing_chunk_size)
            
            concat_datasets = ConcatDataset(datasets)
            return concat_datasets, tensor_names

    def _load_data(self):
        logger.info("\nLoading data into the data silo ..."
                    "{}".format(TRACTOR_SMALL))
        # train data
        train_file = os.path.join(self.processor.data_dir, self.processor.train_filename)
        logger.info("Loading train set from: {} ".format(train_file))
        self.data["train"], self.tensor_names = self._get_dataset(train_file)

        # dev data
        if not self.processor.dev_filename:
            if self.processor.dev_split > 0.0:
                logger.info("Loading dev set as a slice of train set")
                self._create_dev_from_train()
            else:
                logger.info("No dev set is being loaded")
                self.data["dev"] = None
        else:
            dev_file = os.path.join(self.processor.data_dir, self.processor.dev_filename)
            logger.info("Loading dev set from: {}".format(dev_file))
            self.data["dev"], _ = self._get_dataset(dev_file)

        # test data
        if self.processor.test_filename:
            test_file = os.path.join(self.processor.data_dir, self.processor.test_filename)
            logger.info("Loading test set from: {}".format(test_file))
            self.data["test"], _ = self._get_dataset(test_file)
        else:
            logger.info("No test set is being loaded")
            self.data["test"] = None

        # derive stats and meta data
        self._calculate_statistics()
        #self.calculate_class_weights()
        self._initialize_data_loaders()

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

        if self.data["dev"] is not None:
            data_loader_dev = NamedDataLoader(
                dataset=self.data["dev"],
                sampler=SequentialSampler(self.data["dev"]),
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_dev = None

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
        # if(isinstance(self.data["train"], Dataset)):
        #     train_dataset, dev_dataset = random_split(self.data["train"], [n_train, n_dev])
        # else:
        train_dataset, dev_dataset = self.random_split_ConcatDataset(self.data["train"], lengths=[n_train, n_dev])
        self.data["train"] = train_dataset
        if(len(dev_dataset) > 0):
            self.data["dev"] = dev_dataset
        else:
            logger.warning("No dev set created. Maybe adjust the dev_split parameter or the multiprocessing chunk size")

        logger.info(
            f"Took {len(dev_dataset)} samples out of train set to create dev set (dev split is roughly {self.processor.dev_split})"
        )

    def random_split_ConcatDataset(self, ds, lengths):
        """
        Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
        Samples inside Concatdataset should already be shuffled

        Arguments:
            ds (Dataset): Dataset to be split
            lengths (sequence): lengths of splits to be produced
        """
        if sum(lengths) != len(ds):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]

        train = ConcatDataset(ds.datasets[:idx_dataset])
        test = ConcatDataset(ds.datasets[idx_dataset:])
        return train, test

    def _calculate_statistics(self,):
        self.counts = {
            "train": len(self.data["train"])
        }

        if self.data["dev"]:
            self.counts["dev"] = len(self.data["dev"])
        else:
            self.counts["dev"] = 0

        if self.data["test"]:
            self.counts["test"] = len(self.data["test"])
        else:
            self.counts["test"] = 0

        seq_lens = []
        for dataset in self.data["train"].datasets:
            train_input_numpy = dataset[:][0].numpy()
            seq_lens.extend(np.sum(train_input_numpy != 0, axis=1))
        max_seq_len = dataset[:][0].shape[1]

        self.clipped = np.mean(np.array(seq_lens) == max_seq_len)
        self.ave_len = np.mean(seq_lens)

        logger.info("Examples in train: {}".format(self.counts["train"]))
        logger.info("Examples in dev  : {}".format(self.counts["dev"]))
        logger.info("Examples in test : {}".format(self.counts["test"]))
        logger.info("")
        logger.info("Max sequence length:     {}".format(max(seq_lens)))
        logger.info("Average sequence length after clipping: {}".format(self.ave_len))
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
                "batch_size": self.batch_size,
                "ave_seq_len": self.ave_len,
                "clipped": self.clipped
            }
        )

    def calculate_class_weights(self, task_name):
        tensor_name = self.processor.tasks[task_name]["label_tensor_name"]
        label_list = self.processor.tasks[task_name]["label_list"]
        tensor_idx = list(self.tensor_names).index(tensor_name)
        # we need at least ONE observation for each label to avoid division by zero in compute_class_weights.
        observed_labels = copy.deepcopy(label_list)
        for dataset in self.data.values():
            if dataset is not None:
                observed_labels += [label_list[x[tensor_idx].item()] for x in dataset]
        #TODO scale e.g. via logarithm to avoid crazy spikes for rare classes
        class_weights = list(compute_class_weight("balanced", np.asarray(label_list), observed_labels))
        return class_weights


    def get_data_loader(self, dataset):
        return self.loaders[dataset]

    def n_samples(self, dataset):
        """
        Returns the number of samples in a given dataset.

        :param dataset: Choose from train, dev or test
        """
        return self.counts[dataset]
