import pytest
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import GermEval14Processor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

import logging


def test_ner(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 8
    evaluate_every = 50
    lang_model = "bert-base-german-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )

    processor = GermEval14Processor(
        tokenizer=tokenizer, max_seq_len=64, data_dir="samples/ner",train_file="train-sample.txt",
        dev_file="dev-sample.txt",test_file=None
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size)
    language_model = Bert.load(lang_model)
    prediction_head = TokenClassificationHead(layer_dims=[768, len(processor.label_list)])

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        n_examples=data_silo.n_samples("train"),
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device,
    )

    save_dir = "testsave/ner"
    model = trainer.train(model)
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
    ]
    model = Inferencer(save_dir)
    result = model.run_inference(dicts=basic_texts)
    assert result[0]["predictions"][0]["context"] == "Tagesspiegel,"
    assert abs(result[0]["predictions"][0]["probability"] - 0.213869) <= 0.0001