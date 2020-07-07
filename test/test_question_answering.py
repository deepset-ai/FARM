import logging
from pathlib import Path
import numpy as np
import pytest
from math import isclose

from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.infer import Inferencer, QAInferencer
from farm.data_handler.inputs import QAInput, Question


def test_training(distil_bert_squad, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    model, processor = distil_bert_squad
    assert type(model) == AdaptiveModel
    assert type(processor) == SquadProcessor


def test_save_load(distil_bert_squad, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    model, processor = distil_bert_squad

    save_dir = Path("testsave/qa")
    model.save(save_dir)
    processor.save(save_dir)

    inferencer = QAInferencer.load(save_dir, batch_size=2, gpu=False, num_processes=0)
    assert inferencer is not None

def test_inference_dicts(bert_base_squad2):
    qa_format_1 = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]
    qa_format_2 = [{"qas":["Who counted the game among the best ever made?"],
                 "context": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                }]

    result1 = bert_base_squad2.inference_from_dicts(dicts=qa_format_1)
    result2 = bert_base_squad2.inference_from_dicts(dicts=qa_format_2)
    assert result1 == result2

def test_inference_objs(bert_base_squad2, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    obj_input = [QAInput(doc_text="Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                         questions=Question("Who counted the game among the best ever made?"),
                         doc_id="best_id_ever")]
    result = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]

    best_pred = result.prediction[0]

    assert best_pred.answer == "GameTrailers"

    best_score_gold = 11.7282
    best_score = best_pred.score
    assert isclose(best_score, best_score_gold, rel_tol=0.0001)

    no_answer_gap_gold = 12.6491
    no_answer_gap = result.no_answer_gap
    assert isclose(no_answer_gap, no_answer_gap_gold, rel_tol=0.0001)


@pytest.mark.parametrize("num_processes", [None], scope="session")
def test_qa_onnx_inference(adaptive_model_qa, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    QA_input = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]

    # Pytorch
    inferencer = adaptive_model_qa
    result = inferencer.inference_from_dicts(dicts=QA_input)[0]

    # ONNX
    onnx_model_export_path = Path("testsave/onnx-export")
    inferencer.model.convert_to_onnx(onnx_model_export_path)
    inferencer = Inferencer.load(model_name_or_path=onnx_model_export_path, task_type="question_answering", num_processes=0)

    result_onnx = inferencer.inference_from_dicts(QA_input)[0]

    for (onnx, regular) in zip(result_onnx["predictions"][0]["answers"][0].items(), result["predictions"][0]["answers"][0].items()):
        # keys
        assert onnx[0] == regular[0]
        # values
        if type(onnx[1]) == float:
            np.testing.assert_almost_equal(onnx[1], regular[1], decimal=4)  # score
        else:
            assert onnx[1] == regular[1]


if(__name__=="__main__"):
    test_training()
    test_save_load()
    test_inference_dicts()
    test_inference_objs()
    test_qa_onnx_inference()
