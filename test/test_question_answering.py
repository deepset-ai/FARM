import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings


def test_qa(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    batch_size = 2
    n_epochs = 1
    evaluate_every = 4
    base_LM_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=base_LM_model, do_lower_case=False
    )
    label_list = ["start_token", "end_token"]
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=20,
        doc_stride=10,
        max_query_length=6,
        train_filename="train-sample.json",
        dev_filename="dev-sample.json",
        test_filename=None,
        data_dir="samples/qa",
        label_list=label_list,
        metric="squad"
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size)
    language_model = LanguageModel.load(base_LM_model)
    prediction_head = QuestionAnsweringHead(layer_dims=[768, len(label_list)])
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=1e-5,
        warmup_proportion=0.2,
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
    model = trainer.train(model)
    save_dir = "testsave/qa"
    model.save(save_dir)
    processor.save(save_dir)

    QA_input = [
        {
            "questions": ["In what country is Normandy"],
            "text": 'The Normans gave their name to Normandy, a region in France.',
        }
    ]

    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=QA_input,use_multiprocessing=False)
    assert isinstance(result[0]["predictions"][0]["answers"][0]["offset_start"],int)

if(__name__=="__main__"):
    test_qa()