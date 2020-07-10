# fmt: off
import logging
import os
import pprint
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.utils import write_squad_predictions
from farm.data_handler.inputs import QAInput, Question
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

    model = QAInferencer.load("deepset/roberta-base-squad2", batch_size=40, gpu=True, task_type="question_answering")

    # OBJ STYLE
    obj_input = [QAInput(doc_text="Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                         questions=Question("Who counted the game among the best ever made?", uid="best_id_ever"))]
    result = model.inference_from_objects(obj_input)[0]
    pprint.pprint(result)
    raise Exception


    # # DICT STYLE
    # dict_input = [
    #         {
    #             "qas": ["Who counted the game among the best ever made?"],
    #             "context":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
    #         }
    # ]
    # result = model.inference_from_dicts(dicts=dict_input, return_json=False)[0]
    # pprint.pprint(result.to_json())
    # raise Exception

    # FILE STYLE
    filename = os.path.join(processor.data_dir, processor.dev_filename)
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
