import logging
import numpy as np

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import RegressionProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import RegressionHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

def test_doc_regression(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "bert-base-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    processor = RegressionProcessor(tokenizer=tokenizer,
                            max_seq_len=8,
                            data_dir="samples/doc_regr",
                            train_filename="train-sample.tsv",
                            dev_filename="test-sample.tsv",
                            test_filename=None,
                            label_column_name="label")

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = Bert.load(lang_model)
    prediction_head = RegressionHead(layer_dims=[768, 1])
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
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

    save_dir = "testsave/doc_regr"
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": "The dress is just fabulous and it totally fits my size. The fabric is of great quality and the seams are really well hidden. I am super happy with this purchase and I am looking forward to trying some more from the same brand."},
        {"text": "it just did not fit right. The top is very thin showing everything."},
    ]

    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    assert isinstance(result[0]["predictions"][0]["pred"], np.float32)

if(__name__=="__main__"):
    test_doc_regression()