import logging
from pathlib import Path

from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds
import torch

def test_processor_saving_loading(caplog):
    if caplog is not None:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir=Path("samples/doc_class"),
                                            train_filename="train-sample.tsv",
                                            dev_filename=None,
                                            test_filename=None,
                                            label_column_name="coarse_label",
                                            dev_split=0.1,
                                            label_list=["OTHER", "OFFENSE"],
                                            metric=["f1_macro"]
                                            )
    dicts = processor.file_to_dicts(file=Path("samples/doc_class/train-sample.tsv"))
    data, tensor_names, _ = processor.dataset_from_dicts(dicts)

    save_dir = Path("testsave/processor")
    processor.save(save_dir)

    processor = processor.load_from_dir(save_dir)
    dicts = processor.file_to_dicts(file=Path("samples/doc_class/train-sample.tsv"))
    data_loaded, tensor_names_loaded, _ = processor.dataset_from_dicts(dicts)

    assert tensor_names == tensor_names_loaded
    for i in range(len(data.tensors)):
        assert torch.all(torch.eq(data.tensors[i], data_loaded.tensors[i]))

if __name__ == "__main__":
    test_processor_saving_loading(None)
