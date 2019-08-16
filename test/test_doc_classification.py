import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

def test_doc_classification(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 8
    evaluate_every = 30
    lang_model = "bert-base-german-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

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

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = Bert.load(lang_model)
    prediction_head = TextClassificationHead(layer_dims=[768, len(processor.label_list)])
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1)

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device)

    model = trainer.train(model)

    save_dir = "testsave/doc_class"
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": "Martin Müller spielt Handball in Berlin."},
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei."},
    ]
    model = Inferencer.load(save_dir)
    result = model.run_inference(dicts=basic_texts)
    assert result[0]["predictions"][0]["label"] == "OTHER"
    assert abs(result[0]["predictions"][0]["probability"] - 0.7) <= 0.1
