from pathlib import Path

from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel


def onnx_runtime_example():
    """
    This example shows conversion of a transformers model from the Model Hub to
    ONNX format & inference using ONNXRuntime.
    """

    model_name_or_path = "deepset/roberta-base-squad2"
    onnx_model_export_path = Path("./roberta-onnx")

    AdaptiveModel.convert_to_onnx(model_name_or_path, onnx_model_export_path, task_type="question_answering")

    # for ONNX models, the Inferencer uses ONNXRuntime under-the-hood
    inferencer = Inferencer.load(model_name_or_path=onnx_model_export_path)

    qa_input = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. "
            "It received perfect scores from major publications such as 1UP.com, Computer and Video Games, "
            "Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators "
            "GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii "
            "version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called "
            "it one of the greatest games ever created.",
        }
    ]

    results = inferencer.inference_from_dicts(qa_input)
    print(results)
    inferencer.close_multiprocessing_pool()


if __name__ == "__main__":
    onnx_runtime_example()
