# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextPairClassificationProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


def text_pair_classification():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_text_pair_classification")

    ##########################
    ########## Settings ######
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 2
    batch_size = 64
    evaluate_every = 500
    lang_model = "bert-base-cased"
    label_list = ["0", "1"]

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset.
    # The TextPairClassificationProcessor expects a csv with columns called "text', "text_b" and "label"
    processor = TextPairClassificationProcessor(tokenizer=tokenizer,
                                                label_list=label_list,
                                                metric="f1_macro",
                                                max_seq_len=128,
                                                train_filename="train.tsv",
                                                dev_filename="dev.tsv",
                                                test_filename=None,
                                                data_dir=Path("../data/asnq_binary"),
                                                delimiter="\t")

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task
    prediction_head = TextClassificationHead(num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("saved_models/text_pair_classification_model")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    #    Add your own text adapted to the dataset you provide
    # For correct Text Pair Classification on raw dictionaries (inference mode), we need to put both
    # texts (text, text_b) into a tuple.
    # See corresponding conversion in the file_to_dicts() method of TextPairClassificationProcessor: https://github.com/deepset-ai/FARM/blob/5ab5b1620cb51ceb874d4b30c887e377ad1a6e9a/farm/data_handler/processor.py#L744
    basic_texts = [
        {"text": ("how many times have real madrid won the champions league in a row",
                  "They have also won the competition the most times in a row, winning it five times from 1956 to 1960")},
        {"text": ("how many seasons of the blacklist are there on netflix", "Retrieved March 27 , 2018 .")},
    ]

    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)

    print(result)
    model.close_multiprocessing_pool()


if __name__ == "__main__":
    text_pair_classification()

# fmt: on
