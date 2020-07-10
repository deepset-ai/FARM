# fmt: off
import logging
import pprint
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NaturalQuestionsProcessor
from farm.data_handler.inputs import Question, QAInput
from farm.file_utils import fetch_archive_from_http
from farm.infer import QAInferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead, TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


def question_answering():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_natural_questions")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = 24

    # Load model
    fetch_archive_from_http("https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/models/roberta-base-squad2-nq.zip", output_dir="../saved_models/farm")
    model = QAInferencer.load(model_name_or_path="../saved_models/farm/roberta-base-squad2-nq", batch_size=batch_size, gpu=True)


    # # Object Style
    # obj_input = [QAInput(doc_text="Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
    #                      questions=Question("Who counted the game among the best ever made?", uid="best_id_ever"))]
    # result = model.inference_from_objects(obj_input)[0]
    # pprint.pprint(result)
    # raise Exception

    # Dict Style
    qa_format_1 = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]
    qa_format_2 = [
        {
            "qas":["Who counted the game among the best ever made?"],
            "context": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                }
    ]

    result1 = model.inference_from_dicts(dicts=qa_format_1, return_json=False) # result is a list of QAPred objects
    result2 = model.inference_from_dicts(dicts=qa_format_2, return_json=False) # result is a list of QAPred objects

    print(f"\nQuestion: Did GameTrailers rated Twilight Princess as one of the best games ever created?"
          f"\nAnswer from model: {result1[0].prediction[0].answer}")

    pprint.pprint(result1[0].to_json())

if __name__ == "__main__":
    question_answering()


