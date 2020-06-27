import logging
import numpy as np
from pathlib import Path
import pytest

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds, initialize_device_settings


def test_prediction_head_load_save_class_weights(tmp_path, caplog=None):
    """This is a regression test for #428 and #422."""
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    batch_size = 1
    lang_model = "bert-base-german-cased"
    data_dir_path = "samples/doc_class"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    tcp_params = dict(tokenizer=tokenizer,
                      max_seq_len=8,
                      data_dir=Path(data_dir_path),
                      train_filename="train-sample.tsv",
                      label_list=["OTHER", "OFFENSE"],
                      metric="f1_macro",
                      dev_filename="test-sample.tsv",
                      test_filename=None,
                      dev_split=0.0,
                      label_column_name="coarse_label")

    processor = TextClassificationProcessor(**tcp_params)

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = LanguageModel.load(lang_model)
    prediction_head = TextClassificationHead(
        num_labels=2,
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    model.save(tmp_path)
    model_loaded = AdaptiveModel.load(tmp_path, device='cpu')
    assert model_loaded is not None

def test_TextClassificationHead_class_weights_dimensions():
    with pytest.raises(ValueError):
        class_wights = np.asarray([[0.4, 0.6], [0.8, 0.2]])
        TextClassificationHead(
            num_labels=2,
            class_weights=class_wights)
