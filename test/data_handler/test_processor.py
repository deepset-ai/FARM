from pathlib import Path
import pytest

from farm.data_handler.processor import TextClassificationProcessor


def test_TextClassificationProcessor__init__task_names_type():
    with pytest.raises(TypeError):
        TextClassificationProcessor(tokenizer=None,
                                    max_seq_len=128,
                                    data_dir=Path("./data/germeval17"),
                                    metric="acc",
                                    label_list=[["a", "b", "c"], ["x", "y", "z"]],
                                    task_names="only_one_task_as_str",  # this param is wrong
                                    label_column_name=["col_1", "col_2"],
                                    )


def test_TextClassificationProcessor__init__label_list_type():
    with pytest.raises(TypeError):
        TextClassificationProcessor(tokenizer=None,
                                    max_seq_len=128,
                                    data_dir=Path("./data/germeval17"),
                                    metric="acc",
                                    label_list="only_a_str",  # this param is wrong
                                    task_names=["task_1", "task_2"],
                                    label_column_name = ["col_1", "col_2"],
                                    )

def test_TextClassificationProcessor__init__label_list__inner_type():
    with pytest.raises(TypeError):
        TextClassificationProcessor(tokenizer=None,
                                    max_seq_len=128,
                                    data_dir=Path("./data/germeval17"),
                                    metric="acc",
                                    label_list=["a", "b", "c"],  # this parm is wrong
                                    task_names=["task_1", "task_2"],
                                    label_column_name=["col_1", "col_2"],
                                    )

def test_TextClassificationProcessor__init__label_column_name_type():
    with pytest.raises(TypeError):
        TextClassificationProcessor(tokenizer=None,
                                    max_seq_len=128,
                                    data_dir=Path("./data/germeval17"),
                                    metric="acc",
                                    label_list=[["a", "b", "c"], ["x", "y", "z"]],
                                    task_names=["task_1", "task_2"],
                                    label_column_name="only a str",  # this param is wrong
                                    )


def test_TextClassificationProcessor__init__list_len():
    with pytest.raises(AttributeError):
        TextClassificationProcessor(tokenizer=None,
                                    max_seq_len=128,
                                    data_dir=Path("./data/germeval17"),
                                    metric="acc",
                                    label_list=[["a", "b", "c"], ["x", "y", "z"]],
                                    task_names=["task_1", "task_2"],
                                    label_column_name=["col_1", "col_2", "col_3"],  # should only be len 2
                                    )
