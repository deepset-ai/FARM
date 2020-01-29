import json
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings




def question_answering():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    with open("/opt/ml/input/config/hyperparameters.json") as f:
        config = json.load(f)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    evaluate_every = config["evaluate_every"]
    base_LM_model = "roberta-base"
    train_filename = config.get("train_filename", "train-v2.0.json")
    dev_filename = config.get("dev_filename", "dev-v2.0.json")
    data_dir = config.get("data_dir", "../data/squad20")
    save_dir = config["save_dir"]

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=base_LM_model, do_lower_case=False
    )
    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=None,
        data_dir=data_dir,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    # NOTE: In FARM, the dev set metrics differ from test set metrics in that they are calculated on a token level instead of a word level
    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(base_LM_model)
    # b) and a prediction head on top that is suited for our task => Question Answering
    prediction_head = QuestionAnsweringHead()

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        schedule_opts={"name": "LinearWarmup", "warmup_proportion": 0.2},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device
    )
    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
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
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path(save_dir)
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    question_answering()
