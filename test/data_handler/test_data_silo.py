from pathlib import Path
import numpy as np

from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo, DataSiloForNestedCrossVal


def test_data_silo_for_nested_cross_val():
    lang_model = "bert-base-german-cased"
    n_outer_splits = 7
    n_inner_splits = 11

    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=64,
                                            data_dir=Path("../data/germeval18"),
                                            label_list=["OTHER", "OFFENSE"],
                                            metric="f1_macro",
                                            label_column_name="coarse_label"
                                            )

    data_silo = DataSilo(processor=processor, batch_size=32)

    silos = DataSiloForNestedCrossVal.make(
        data_silo,
        sets=['test', 'train'],
        n_outer_splits=n_outer_splits,
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
