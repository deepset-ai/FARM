from pathlib import Path
import numpy as np

from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal, DataSiloForHoldout


def test_data_silo_for_cross_val_nested():
    lang_model = "distilbert-base-german-cased"
    n_outer_splits = 3
    n_inner_splits = 3

    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=64,
                                            data_dir=Path("data/germeval18"),
                                            label_list=["OTHER", "OFFENSE"],
                                            metric="f1_macro",
                                            label_column_name="coarse_label"
                                            )

    data_silo = DataSilo(processor=processor, batch_size=32)

    silos = DataSiloForCrossVal.make(
        data_silo,
        sets=['test', 'train'],
        n_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        )

    # check number of silos
    assert len(silos) == (n_outer_splits * n_inner_splits)

    # because the outer cross validation creates the test set it must be the same
    # in silo 0 and silo 1
    data_loader_test_indices_0 = silos[0].get_data_loader('test').dataset.indices
    data_loader_test_indices_1 = silos[1].get_data_loader('test').dataset.indices
    assert data_loader_test_indices_0.size > 0
    assert data_loader_test_indices_1.size > 0
    assert data_loader_test_indices_0.ndim == 1
    assert data_loader_test_indices_1.ndim == 1
    assert np.array_equal(data_loader_test_indices_0, data_loader_test_indices_1)

    # because the inner cross validation creates the dev set it must be different
    # in silo 0 and silo 1
    data_loader_dev_indices_0 = silos[0].get_data_loader('dev').dataset.indices
    data_loader_dev_indices_1 = silos[1].get_data_loader('dev').dataset.indices
    assert data_loader_dev_indices_0.size > 0
    assert data_loader_dev_indices_1.size > 0
    assert data_loader_dev_indices_0.ndim == 1
    assert data_loader_dev_indices_1.ndim == 1
    assert not np.array_equal(data_loader_dev_indices_0, data_loader_dev_indices_1)

    # extract and test train sets of silo 0 and 1
    data_loader_train_indices_0 = silos[0].get_data_loader('train').dataset.indices
    data_loader_train_indices_1 = silos[1].get_data_loader('train').dataset.indices
    assert data_loader_train_indices_0.size > 0
    assert data_loader_train_indices_1.size > 0
    assert data_loader_train_indices_0.ndim == 1
    assert data_loader_train_indices_1.ndim == 1

    # size of dev + train + test must be same on all folds
    assert (data_loader_train_indices_0.size +
           data_loader_dev_indices_0.size +
           data_loader_test_indices_0.size) == \
           (data_loader_train_indices_1.size +
           data_loader_dev_indices_1.size +
           data_loader_test_indices_1.size)

    del tokenizer
    del processor
    del data_silo
    del silos


def test_data_silo_for_cross_val1():
    n_splits = 5
    lang_model = "distilbert-base-german-cased"
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model)
    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=64,
                                            data_dir=Path("data/germeval18"),
                                            label_list=["OTHER", "OFFENSE"],
                                            metric="f1_macro",
                                            dev_split=0.2,
                                            dev_stratification=False,
                                            label_column_name="coarse_label"
                                            )
    data_silo = DataSilo(processor=processor, batch_size=32, max_processes=128)
    silos = DataSiloForCrossVal.make(
        data_silo,
        sets=['train'],
        n_splits=n_splits,
        )

    assert len(silos) == n_splits
    for idx, silo in enumerate(silos):
        train_ds = silo.get_data_loader("train").dataset
        dev_ds = silo.get_data_loader("dev").dataset
        test_ds = silo.get_data_loader("test").dataset
        train_indices = train_ds.datasets[0].indices
        dev_indices = dev_ds.indices
        train_idx_set = set([i for i in train_indices])
        dev_idx_set = set([i for i in dev_indices])
        orig_train = train_ds.datasets[0].dataset
        orig_train_idx = orig_train.indices
        test_idx = test_ds.indices
        orig_train_idx_set = set([i for i in orig_train_idx])
        test_idx_set = set([i for i in test_idx])
        assert len(orig_train_idx_set.intersection(test_idx_set)) == 0
        assert len(train_idx_set.intersection(dev_idx_set)) == 0
    del tokenizer
    del processor
    del data_silo
    del silos


def test_data_silo_for_holdout1():
    n_splits = 5
    lang_model = "distilbert-base-german-cased"
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model)
    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=64,
                                            data_dir=Path("data/germeval18"),
                                            label_list=["OTHER", "OFFENSE"],
                                            metric="f1_macro",
                                            dev_split=0.2,
                                            dev_stratification=False,
                                            label_column_name="coarse_label"
                                            )
    data_silo = DataSilo(processor=processor, batch_size=32, max_processes=128)
    silos = DataSiloForHoldout.make(
        data_silo,
        sets=['train'],
        n_splits=n_splits,
        train_split=0.8,
        )

    assert len(silos) == n_splits
    for idx, silo in enumerate(silos):
        train_ds = silo.get_data_loader("train").dataset
        dev_ds = silo.get_data_loader("dev").dataset
        test_ds = silo.get_data_loader("test").dataset
        train_indices = train_ds.datasets[0].indices
        dev_indices = dev_ds.indices
        train_idx_set = set([i for i in train_indices])
        dev_idx_set = set([i for i in dev_indices])
        orig_train = train_ds.datasets[0].dataset
        orig_train_idx = orig_train.indices
        test_idx = test_ds.indices
        orig_train_idx_set = set([i for i in orig_train_idx])
        test_idx_set = set([i for i in test_idx])
        assert len(orig_train_idx_set.intersection(test_idx_set)) == 0
        assert len(train_idx_set.intersection(dev_idx_set)) == 0
    del tokenizer
    del processor
    del data_silo
    del silos