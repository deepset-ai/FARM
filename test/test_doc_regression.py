import logging
from pathlib import Path

import numpy as np
import pytest

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import RegressionProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import RegressionHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

@pytest.mark.parametrize("data_dir_path,text_column_name",
                         [("samples/doc_regr", None),
                          ("samples/doc_regr_other_text_column_name", "text_other")])
def test_doc_regression(data_dir_path, text_column_name, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    rp_params = dict(tokenizer=tokenizer,
                            max_seq_len=8,
                            data_dir=Path(data_dir_path),
                            train_filename="train-sample.tsv",
                            dev_filename="test-sample.tsv",
                            test_filename=None,
                            label_column_name="label")

    if text_column_name is not None:
        rp_params["text_column_name"] = text_column_name

    processor = RegressionProcessor(**rp_params)

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = LanguageModel.load(lang_model)
    prediction_head = RegressionHead()
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
        device=device
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1,
        device=device,
        schedule_opts={'name': 'CosineWarmup', 'warmup_proportion': 0.1}
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device
    )

    trainer.train()

    save_dir = Path("testsave/doc_regr")
    model.save(save_dir)
    processor.save(save_dir)

    del model
    del processor
    del optimizer
    del data_silo
    del trainer

    basic_texts = [
        {"text": "The dress is just fabulous and it totally fits my size. The fabric is of great quality and the seams are really well hidden. I am super happy with this purchase and I am looking forward to trying some more from the same brand."},
        {"text": "it just did not fit right. The top is very thin showing everything."},
    ]

    model = Inferencer.load(save_dir, num_processes=0)
    result = model.inference_from_dicts(dicts=basic_texts)
    assert isinstance(result[0]["predictions"][0]["pred"], np.float32)
    del model
