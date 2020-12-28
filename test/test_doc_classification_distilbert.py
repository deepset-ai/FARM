from pathlib import Path
import logging
import numpy as np

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import DistilBert
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings


def test_doc_classification(caplog):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "distilbert-base-german-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=8,
                                            data_dir=Path("samples/doc_class"),
                                            train_filename=Path("train-sample.tsv"),
                                            label_list=["OTHER", "OFFENSE"],
                                            metric="f1_macro",
                                            dev_filename="test-sample.tsv",
                                            test_filename=None,
                                            dev_split=0.0,
                                            label_column_name="coarse_label")

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = DistilBert.load(lang_model)
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

    save_dir = Path("testsave/doc_class")
    model.save(save_dir)
    processor.save(save_dir)

    del model
    del processor
    del optimizer
    del data_silo
    del trainer

    basic_texts = [
        {"text": "Malte liebt Berlin."},
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei."}
    ]

    inf = Inferencer.load(save_dir, batch_size=2, num_processes=0)
    result = inf.inference_from_dicts(dicts=basic_texts)
    assert isinstance(result[0]["predictions"][0]["probability"], np.float32)
    del inf

if __name__ == "__main__":
    test_doc_classification(None)
