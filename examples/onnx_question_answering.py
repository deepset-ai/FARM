from pathlib import Path

from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel


def onnx_runtime_example():
    """
    This example converts a Question Answering FARM AdaptiveModel
    to ONNX format and uses ONNX Runtime for doing Inference.
    """

    device = "cpu"
    model_name_or_path = "deepset/bert-base-cased-squad2"
    onnx_model_export_path = Path("./onnx-export")

    model = AdaptiveModel.convert_from_transformers(model_name_or_path, device=device, task_type="question_answering")
    model.convert_to_onnx(onnx_model_export_path)

    inferencer = Inferencer.load(model_name_or_path=onnx_model_export_path)

    qa_input = [
        {
            "qas": ["Who counted the game among the best ever made?"],
            "context": "Twilight Princess was released to universal critical acclaim and commercial success. "
            "It received perfect scores from major publications such as 1UP.com, Computer and Video Games, "
            "Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators "
            "GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii "
            "version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called "
            "it one of the greatest games ever created.",
        }
    ]

    results = inferencer.inference_from_dicts(qa_input)
    print(results)


if __name__ == "__main__":
    onnx_runtime_example()
