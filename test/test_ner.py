from pathlib import Path
import pytest

import numpy as np

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

import logging

# TODO: Test slow tokenizers when reimplemented
@pytest.mark.parametrize("use_fast", [True])
def test_ner(caplog, use_fast):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 3
    batch_size = 2
    evaluate_every = 1
    lang_model = "distilbert-base-german-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=False,
        use_fast=use_fast,
    )

    ner_labels = ["[PAD]", "X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH",
                  "I-OTH"]

    processor = NERProcessor(
        tokenizer=tokenizer,
        max_seq_len=8,
        data_dir=Path("samples/ner"),
        train_filename="train-sample.txt",
        dev_filename="dev-sample.txt",
        test_filename=None,
        delimiter=" ",
        label_list=ner_labels,
        metric="seq_f1",
        multithreading_rust=False
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)
    language_model = LanguageModel.load(lang_model)
    prediction_head = TokenClassificationHead(num_labels=13)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1,
        device=device,
        schedule_opts={'name': 'LinearWarmup', 'warmup_proportion': 0.1}
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )

    save_dir = Path("testsave/ner")
    model = trainer.train()
    model.save(save_dir)
    processor.save(save_dir)

    del model
    del processor
    del optimizer
    del data_silo
    del trainer

    basic_texts = [
        {"text": "Paris is a town in France."},
    ]
    model = Inferencer.load(model_name_or_path="dbmdz/bert-base-cased-finetuned-conll03-english", num_processes=0, task_type="ner", use_fast=use_fast)
    # labels arent correctly inserted from transformers
    # They are converted to LABEL_1 ... LABEL_N
    # For the inference result to contain predictions we need them in IOB NER format
    model.processor.tasks["ner"]["label_list"][-1] = "B-LOC"
    result = model.inference_from_dicts(dicts=basic_texts)

    assert result[0]["predictions"][0]["context"] == "Paris"
    assert isinstance(result[0]["predictions"][0]["probability"], np.float32)


if __name__ == "__main__":
    test_ner(None, True)
