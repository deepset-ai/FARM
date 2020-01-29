import copy
import logging
import torch.multiprocessing as mp
import os
from contextlib import ExitStack
from functools import partial
import random
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.processor import Processor
from farm.data_handler.utils import grouper
from farm.utils import MLFlowLogger as MlLogger
from farm.utils import log_ascii_workers, calc_chunksize
from farm.utils import get_dict_checksum
from farm.visual.ascii.images import TRACTOR_SMALL

logger = logging.getLogger(__name__)


class DataSilo:
    """ Generates and stores PyTorch DataLoader objects for the train, dev and test datasets.
    Relies upon functionality in the processor to do the conversion of the data. Will also
    calculate and display some statistics.
     """

    def __init__(
        self,
        processor,
        batch_size,
        distributed=False,
        automatic_loading=True,
        max_multiprocessing_chunksize=2000,
        max_processes=128,
        caching=False,
        cache_path=Path("cache/data_silo"),
    ):
        """
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :type processor: Processor
        :param batch_size: The size of batch that should be returned by the DataLoaders.
        :type batch_size: int
        :param distributed: Set to True if the program is running in a distributed setting.
        :type distributed: bool
        :param automatic_loading: Set to False, if you don't want to automatically load data at initialization.
        :type automatic_loading: bool
        :param max_multiprocessing_chunksize: max possible value for chunksize as calculated by `calc_chunksize()`
            in `farm.utils`. For certain cases like lm_finetuning, a smaller value can be set, as the default chunksize
            values are rather large that might cause memory issues.
        :type max_multiprocessing_chunksize: int
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing ot make debugging easier.
        :type max_processes: int
        :param caching: save the processed datasets on disk to save time/compute if the same train data is used to run
                        multiple experiments. Each cache has a checksum based on the train_filename of the Processor
                        and the batch size.
        :type caching: bool
        :param cache_path: root dir for storing the datasets' cache.
        :type cache_path: Path
        """
        self.distributed = distributed
        self.processor = processor
        self.data = {}
        self.batch_size = batch_size
        self.class_weights = None
        self.max_processes = max_processes
        self.max_multiprocessing_chunksize = max_multiprocessing_chunksize
        self.caching = caching
        self.cache_path = cache_path

        loaded_from_cache = False
        if self.caching:  # Check if DataSets are present in cache
            checksum = self._get_checksum()
            dataset_path = self.cache_path / checksum

            if dataset_path.exists():
                self._load_dataset_from_cache(dataset_path)
                loaded_from_cache = True

        if not loaded_from_cache and automatic_loading:
            # In most cases we want to load all data automatically, but in some cases we rather want to do this
            # later or load from dicts instead of file (https://github.com/deepset-ai/FARM/issues/85)
            self._load_data()

    @classmethod
    def _dataset_from_chunk(cls, chunk, processor):
        """
        Creating a dataset for a chunk (= subset) of dicts. In multiprocessing:
          * we read in all dicts from a file
          * split all dicts into chunks
          * feed *one chunk* to *one process*
          => the *one chunk*  gets converted to *one dataset* (that's what we do here)
          * all datasets get collected and concatenated
        :param chunk: Instead of only having a list of dicts here we also supply an index (ascending int) for each.
            => [(0, dict), (1, dict) ...]
        :type chunk: list of tuples
        :param processor: FARM Processor (e.g. TextClassificationProcessor)
        :return: PyTorch Dataset
        """
        dicts = [d[1] for d in chunk]
        indices = [x[0] for x in chunk]
        dataset = processor.dataset_from_dicts(dicts=dicts, indices=indices)
        return dataset

    def _get_dataset(self, filename, dicts=None):
        if not filename and not dicts:
            raise ValueError("You must either supply `filename` or `dicts`")

        # loading dicts from file (default)
        if dicts is None:
            dicts = self.processor.file_to_dicts(filename)
            #shuffle list of dicts here if we later want to have a random dev set splitted from train set
            if str(self.processor.train_filename) in str(filename):
                if not self.processor.dev_filename:
                    if self.processor.dev_split > 0.0:
                        random.shuffle(dicts)

        num_dicts = len(dicts)
        multiprocessing_chunk_size, num_cpus_used = calc_chunksize(
            num_dicts=num_dicts,
            max_processes=self.max_processes,
            max_chunksize=self.max_multiprocessing_chunksize,
        )

        with ExitStack() as stack:
            if self.max_processes > 1:  # use multiprocessing only when max_processes > 1
                p = stack.enter_context(mp.Pool(processes=num_cpus_used))

                logger.info(
                    f"Got ya {num_cpus_used} parallel workers to convert {num_dicts} dictionaries "
                    f"to pytorch datasets (chunksize = {multiprocessing_chunk_size})..."
                )
                log_ascii_workers(num_cpus_used, logger)

                results = p.imap(
                    partial(self._dataset_from_chunk, processor=self.processor),
                    grouper(dicts, multiprocessing_chunk_size),
                    chunksize=1,
                )
            else:
                logger.info(
                    f"Multiprocessing disabled, using a single worker to convert {num_dicts}"
                    f"dictionaries to pytorch datasets."
                )

                results = map(partial(self._dataset_from_chunk, processor=self.processor), grouper(dicts, num_dicts))

            datasets = []

            with tqdm(total=len(dicts), unit=' Dicts', desc="Preprocessing Dataset") as pbar:
                for dataset, tensor_names in results:
                    datasets.append(dataset)
                    pbar.update(multiprocessing_chunk_size)

            concat_datasets = ConcatDataset(datasets)
            return concat_datasets, tensor_names

    def _load_data(self, train_dicts=None, dev_dicts=None, test_dicts=None):
        """
        Loading the train, dev and test datasets either from files (default) or from supplied dicts.
        The processor is called to handle the full conversion from "raw data" to a Pytorch Dataset.
        The resulting datasets are loaded into DataSilo.data

        :param train_dicts: (Optional) dicts containing examples for training.
        :param dev_dicts: (Optional) dicts containing examples for dev.
        :param test_dicts: (Optional) dicts containing examples for test.
        :return: None
        """
        logger.info("\nLoading data into the data silo ..."
                    "{}".format(TRACTOR_SMALL))
        # train data
        if train_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["train"], self.tensor_names = self._get_dataset(filename=None, dicts=train_dicts)
        else:
            # or from a file (default)
            train_file = self.processor.data_dir / self.processor.train_filename
            logger.info("Loading train set from: {} ".format(train_file))
            self.data["train"], self.tensor_names = self._get_dataset(train_file)

        # dev data
        if dev_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["dev"], self.tensor_names = self._get_dataset(filename=None, dicts=dev_dicts)
        elif self.processor.dev_filename:
            # or from file (default)
            dev_file = self.processor.data_dir / self.processor.dev_filename
            logger.info("Loading dev set from: {}".format(dev_file))
            self.data["dev"], _ = self._get_dataset(dev_file)
        elif self.processor.dev_split > 0.0:
            # or split it apart from train set
            logger.info("Loading dev set as a slice of train set")
            self._create_dev_from_train()
        else:
            logger.info("No dev set is being loaded")
            self.data["dev"] = None

        # test data
        if test_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["test"], self.tensor_names = self._get_dataset(filename=None, dicts=test_dicts)
        elif self.processor.test_filename:
            # or from file (default)
            test_file = self.processor.data_dir / self.processor.test_filename
            logger.info("Loading test set from: {}".format(test_file))
            self.data["test"], _ = self._get_dataset(test_file)
        else:
            logger.info("No test set is being loaded")
            self.data["test"] = None

        if self.caching:
            self._save_dataset_to_cache()

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _load_dataset_from_cache(self, cache_dir):
        """
        Load serialized dataset from a cache.
        """
        logger.info(f"Loading datasets from cache at {cache_dir}")
        self.data["train"] = torch.load(cache_dir / "train_dataset")

        dev_dataset_path = cache_dir / "dev_dataset"
        if dev_dataset_path.exists():
            self.data["dev"] = torch.load(dev_dataset_path)
        else:
            self.data["dev"] = None

        test_dataset_path = cache_dir / "test_dataset"
        if test_dataset_path.exists():
            self.data["test"] = torch.load(test_dataset_path)
        else:
            self.data["test"] = None

        self.tensor_names = torch.load(cache_dir / "tensor_names")

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _get_checksum(self):
        """
        Get checksum based on a dict to ensure validity of cached DataSilo
        """
        # keys in the dict identifies uniqueness for a given DataSilo.
        payload_dict = {
            "train_filename": str(Path(self.processor.train_filename).absolute()),
            "data_dir": str(self.processor.data_dir.absolute()),
            "max_seq_len": self.processor.max_seq_len,
            "dev_split": self.processor.dev_split,
            "tasks": self.processor.tasks
        }
        checksum = get_dict_checksum(payload_dict)
        return checksum

    def _save_dataset_to_cache(self):
        """
        Serialize and save dataset to a cache.
        """
        checksum = self._get_checksum()

        cache_dir = self.cache_path / checksum
        cache_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.data["train"], cache_dir / "train_dataset")

        if self.data["dev"]:
            torch.save(self.data["dev"], cache_dir / "dev_dataset")

        if self.data["test"]:
            torch.save(self.data["test"], cache_dir / "test_dataset")

        torch.save(self.tensor_names, cache_dir / "tensor_names")
        logger.info(f"Cached the datasets at {cache_dir}")

    def _initialize_data_loaders(self):
        """ Initializing train, dev and test data loaders for the already loaded datasets """

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
        """ Split a dev set apart from the train dataset """
        n_dev = int(self.processor.dev_split * len(self.data["train"]))
        n_train = len(self.data["train"]) - n_dev

        train_dataset, dev_dataset = self.random_split_ConcatDataset(self.data["train"], lengths=[n_train, n_dev])
        self.data["train"] = train_dataset
        if(len(dev_dataset) > 0):
            self.data["dev"] = dev_dataset
        else:
            logger.warning("No dev set created. Please adjust the dev_split parameter.")

        logger.info(
            f"Took {len(dev_dataset)} samples out of train set to create dev set (dev split is roughly {self.processor.dev_split})"
        )

    def random_split_ConcatDataset(self, ds, lengths):
        """
        Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
        Samples inside Concatdataset should already be shuffled

        :param ds: Dataset to be split
        :type ds: Dataset
        :param lengths: lengths of splits to be produced
        :type lengths: list
        """
        if sum(lengths) != len(ds):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        try:
            idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
        except IndexError:
            raise Exception("All dataset chunks are being assigned to train set leaving no samples for dev set. "
                            "Either consider increasing dev_split or setting it to 0.0\n"
                            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
                            f"train/dev split: {lengths}")

        assert idx_dataset >= 1, "Dev_split ratio is too large, there is no data in train set. " \
                             f"Please lower dev_split = {self.processor.dev_split}"

        train = ConcatDataset(ds.datasets[:idx_dataset])
        test = ConcatDataset(ds.datasets[idx_dataset:])
        return train, test

    def _calculate_statistics(self):
        """ Calculate and log simple summary statistics of the datasets """

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
            seq_lens.extend(np.sum(train_input_numpy != self.processor.tokenizer.pad_token_id, axis=1))
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

    def calculate_class_weights(self, task_name, source="train"):
        """ For imbalanced datasets, we can calculate class weights that can be used later in the
        loss function of the prediction head to upweight the loss of minorities.

        :param task_name: name of the task as used in the processor
        :type task_name: str
        """

        tensor_name = self.processor.tasks[task_name]["label_tensor_name"]
        label_list = self.processor.tasks[task_name]["label_list"]
        tensor_idx = list(self.tensor_names).index(tensor_name)
        # we need at least ONE observation for each label to avoid division by zero in compute_class_weights.
        observed_labels = copy.deepcopy(label_list)
        if source == "all":
            datasets = self.data.values()
        elif source == "train":
            datasets = [self.data["train"]]
        else:
            raise Exception("source argument expects one of [\"train\", \"all\"]")
        for dataset in datasets:
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


class DataSiloForCrossVal:
    """
    For performing cross validation, we really want to combine all the instances from all
    the sets or just some of the sets, then create a different data silo instance for each fold.
    Calling DataSiloForCrossVal.make() creates a list of DataSiloForCrossVal instances - one for each fold.
    """

    def __init__(self, origsilo, trainset, devset, testset):
        self.tensor_names = origsilo.tensor_names
        self.data = {"train": trainset, "dev": devset, "test": testset}
        self.processor = origsilo.processor
        self.batch_size = origsilo.batch_size
        # should not be necessary, xval makes no sense with huge data
        # sampler_train = DistributedSampler(self.data["train"])
        sampler_train = RandomSampler(trainset)

        self.data_loader_train = NamedDataLoader(
            dataset=trainset,
            sampler=sampler_train,
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.data_loader_dev = NamedDataLoader(
            dataset=devset,
            sampler=SequentialSampler(devset),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.data_loader_test = NamedDataLoader(
            dataset=testset,
            sampler=SequentialSampler(testset),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.loaders = {
            "train": self.data_loader_train,
            "dev": self.data_loader_dev,
            "test": self.data_loader_test,
        }

    def get_data_loader(self, which):
        return self.loaders[which]

    @staticmethod
    def make(datasilo, sets=["train", "dev", "test"], n_splits=5, stratified=True,
             shuffle=True, random_state=None, dev_split=0.2):
        """
        Create number of folds data-silo-like objects which can be used for training from the
        original data silo passed on.

        :param datasilo: the data silo that contains the original data
        :param sets: which sets to use to create the xval folds
        :param n_splits: number of folds to create
        :param stratified: if class stratificiation should be done
        :param shuffle: shuffle each class' samples before splitting
        :param random_state: random state for shuffling
        :param dev_split: size of the dev set for a fold, held out from the training set
        """
        setstoconcat = [datasilo.data[setname] for setname in sets]
        ds_all = ConcatDataset(setstoconcat)
        idxs = list(range(len(ds_all)))
        if stratified:
            # get all the labels for stratification
            ytensors = [t[3][0] for t in ds_all]
            Y = torch.stack(ytensors)
            xval = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            xval_split = xval.split(idxs,Y)
        else:
            xval = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            xval_split = xval.split(idxs)
        # for each fold create a DataSilo4Xval instance, where the training set is further
        # divided into actual train and dev set
        silos = []
        for train_idx, test_idx in xval_split:
            n_dev = int(dev_split * len(train_idx))
            n_actual_train = len(train_idx) - n_dev
            # TODO: this split into actual train and test set could/should also be stratified, for now
            # we just do this by taking the first/last indices from the train set (which should be
            # shuffled by default)
            actual_train_idx = train_idx[:n_actual_train]
            dev_idx = train_idx[n_actual_train:]
            # create the actual datasets
            ds_train = Subset(ds_all, actual_train_idx)
            ds_dev = Subset(ds_all, dev_idx)
            ds_test = Subset(ds_all, test_idx)
            silos.append(DataSiloForCrossVal(datasilo, ds_train, ds_dev, ds_test))
        return silos