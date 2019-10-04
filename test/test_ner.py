import numpy as np

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.modeling.optimization import initialize_optimizer
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
    batch_size = 2
    evaluate_every = 1
    lang_model = "bert-base-german-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )

    ner_labels = ["[PAD]", "X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH",
                  "I-OTH"]

    processor = NERProcessor(
        tokenizer=tokenizer, max_seq_len=8, data_dir="samples/ner",train_filename="train-sample.txt",
        dev_filename="dev-sample.txt",test_filename=None, delimiter=" ", label_list=ner_labels, metric="seq_f1"
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size)
    language_model = Bert.load(lang_model)
    prediction_head = TokenClassificationHead(layer_dims=[768, len(ner_labels)])

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
        n_batches=len(data_silo.loaders["train"]),
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
    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    assert result[0]["predictions"][0]["context"] == "sagte"
    assert isinstance(result[0]["predictions"][0]["probability"], np.float32)


if(__name__=="__main__"):
    test_ner()