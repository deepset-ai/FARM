# fmt: off
import logging
import pprint

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings


def test_qa(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    batch_size = 6
    n_epochs = 2
    evaluate_every = 100
    base_LM_model = "bert-base-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_LM_model, do_lower_case=False
    )
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=64,
        train_filename="train-sample.json",
        dev_filename="dev-sample.json",
        test_filename=None,
        data_dir="samples/qa",
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)
    language_model = Bert.load(base_LM_model)
    prediction_head = QuestionAnsweringHead(layer_dims=[768, len(processor.label_list)])
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
    model = trainer.train(model)
    save_dir = "testsave/qa"
    model.save(save_dir)
    processor.save(save_dir)

    QA_input = [
            {
                "questions": ["In what country is Normandy located?"],
                "text":  "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
            }]

    model = Inferencer(save_dir)
    result = model.run_inference(dicts=QA_input)
    assert result[0]["predictions"][0]["label"] == 'The'
