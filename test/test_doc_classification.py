import logging
from pathlib import Path

import numpy as np
import pytest

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings


@pytest.mark.parametrize("data_dir_path,text_column_name",
                         [("samples/doc_class", None),
                          ("samples/doc_class_other_text_column_name", "text_other")])
@pytest.mark.parametrize("use_fast", [False, True])
def test_doc_classification(data_dir_path, text_column_name, use_fast, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "bert-base-german-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False,
        use_fast=use_fast,
        )

    tcp_params = dict(tokenizer=tokenizer,
                      max_seq_len=8,
                      data_dir=Path(data_dir_path),
                      train_filename="train-sample.tsv",
                      label_list=["OTHER", "OFFENSE"],
                      metric="f1_macro",
                      dev_filename="test-sample.tsv",
                      test_filename="test-sample.tsv",
                      dev_split=0.0,
                      label_column_name="coarse_label")

    if text_column_name is not None:
        tcp_params["text_column_name"] = text_column_name

    processor = TextClassificationProcessor(**tcp_params)

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = LanguageModel.load(lang_model)
    prediction_head = TextClassificationHead(num_labels=2)
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1,
        device=device,
        schedule_opts=None)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    trainer.train()

    save_dir = Path("testsave/doc_class_bert")
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": "Martin Müller spielt Handball in Berlin."},
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei."}
    ]

    inf = Inferencer.load(save_dir, batch_size=2, num_processes=0)
    result = inf.inference_from_dicts(dicts=basic_texts)
    assert isinstance(result[0]["predictions"][0]["probability"], np.float32)
    result2 = inf.inference_from_dicts(dicts=basic_texts, return_json=True)
    assert result == result2

    # is the rest result stored?
    assert trainer.test_result is not None
