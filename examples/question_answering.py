# fmt: off
import logging
import os
import pprint
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.utils import write_squad_predictions
from farm.infer import QAInferencer
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

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = 24
    n_epochs = 2
    evaluate_every = 2000
    lang_model = "roberta-base"
    do_lower_case = False # roberta is a cased model
    train_filename = "train-v2.0.json"
    dev_filename = "dev-v2.0.json"

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
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
        data_dir=Path("../data/squad20"),
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    # NOTE: In FARM, the dev set metrics differ from test set metrics in that they are calculated on a token level instead of a word level
    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
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
    save_dir = Path("../saved_models/bert-english-qa-tutorial")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    QA_input = [
            {
                "questions": ["Who counted the game among the best ever made?"],
                "text":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
            }]

    model = QAInferencer.load(save_dir, batch_size=40, gpu=True)
    result = model.inference_from_dicts(dicts=QA_input)[0]

    pprint.pprint(result)
    model.close_multiprocessing_pool()

    # 10. Do Inference on whole SQuAD Dataset & write the predictions file to disk
    filename = os.path.join(processor.data_dir,processor.dev_filename)
    result = model.inference_from_file(file=filename, return_json=False)
    result_squad = [x.to_squad_eval() for x in result]

    write_squad_predictions(
        predictions=result_squad,
        predictions_filename=filename,
        out_filename="predictions.json"
    )

    # 11. Get final evaluation metric using the official SQuAD evaluation script
    # To evaluate the model's performance on the SQuAD dev set, run the official squad eval script
    # (farm/squad_evaluation.py) in the command line with something like the command below.
    # This is necessary since the FARM evaluation during training is done on the token level.
    # This script performs word level evaluation and will generate metrics that are comparable
    # to the SQuAD leaderboard and most other frameworks:
    #       python squad_evaluation.py path/to/squad20/dev-v2.0.json path/to/predictions.json

if __name__ == "__main__":
    question_answering()
