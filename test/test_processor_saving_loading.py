import logging
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.tokenization import BertTokenizer
from farm.utils import set_all_seeds
import torch

def test_processor_saving_loading(caplog):
    caplog.set_level(logging.CRITICAL)
    set_all_seeds(seed=42)
    lang_model = "bert-base-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model, do_lower_case=False, never_split_chars=["-", "_"]
    )

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir="samples/doc_class",
                                            train_filename="train-sample.tsv",
                                            dev_filename=None,
                                            test_filename=None,
                                            dev_split=0.1,
                                            columns=["text", "label", "unused"],
                                            label_list=["OTHER", "OFFENSE"],
                                            metrics=["f1_macro"]
                                            )
    dicts = processor.file_to_dicts(file="samples/doc_class/train-sample.tsv")
    data, tensor_names = processor.dataset_from_dicts(dicts)

    save_dir = "testsave/processor"
    processor.save(save_dir)

    processor = processor.load_from_dir(save_dir)
    dicts = processor.file_to_dicts(file="samples/doc_class/train-sample.tsv")
    data_loaded, tensor_names_loaded = processor.dataset_from_dicts(dicts)

    assert tensor_names == tensor_names_loaded
    for i in range(len(data.tensors)):
        assert torch.all(torch.eq(data.tensors[i], data_loaded.tensors[i]))
