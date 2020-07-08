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


@pytest.fixture()
def span_inference_result(bert_base_squad2, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    obj_input = [QAInput(doc_text="Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                         questions=Question("Who counted the game among the best ever made?", uid="best_id_ever"))]
    result = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]
    return result


@pytest.fixture()
def no_answer_inference_result(bert_base_squad2, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    obj_input = [QAInput(doc_text="The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.",
                         questions=Question("The Amazon represents less than half of the planets remaining what?", uid="best_id_ever"))]
    result = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]
    return result


def test_inference_objs(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    assert span_inference_result


def test_span_performance(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    best_pred = span_inference_result.prediction[0]

    assert best_pred.answer == "GameTrailers"

    best_score_gold = 11.7282
    best_score = best_pred.score
    assert isclose(best_score, best_score_gold, rel_tol=0.0001)

    no_answer_gap_gold = 12.6491
    no_answer_gap = span_inference_result.no_answer_gap
    assert isclose(no_answer_gap, no_answer_gap_gold, rel_tol=0.0001)


def test_no_answer_performance(no_answer_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    best_pred = no_answer_inference_result.prediction[0]

    assert best_pred.answer == "no_answer"

    best_score_gold = 15.8022
    best_score = best_pred.score
    assert isclose(best_score, best_score_gold, rel_tol=0.0001)

    no_answer_gap_gold = -15.0159
    no_answer_gap = no_answer_inference_result.no_answer_gap
    assert isclose(no_answer_gap, no_answer_gap_gold, rel_tol=0.0001)


def test_qa_pred_attributes(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    qa_pred = span_inference_result
    attributes_gold = ['aggregation_level', 'answer_types', 'answers_to_json', 'context', 'context_window_size',
                       'create_context', 'ground_truth_answer', 'id', 'n_passages', 'no_answer_gap', 'prediction',
                       'question', 'to_json', 'to_squad_eval', 'token_offsets']
    for ag in attributes_gold:
        assert ag in dir(qa_pred)


def test_qa_candidate_attributes(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    qa_candidate = span_inference_result.prediction[0]
    attributes_gold = ['add_answer', 'add_cls', 'aggregation_level', 'answer', 'answer_support', 'answer_type',
                       'context', 'n_passages_in_doc', 'offset_answer_end', 'offset_answer_start',
                       'offset_answer_support_end', 'offset_answer_support_start', 'offset_context_end',
                       'offset_context_start', 'offset_unit', 'passage_id', 'probability', 'score', 'span_to_string',
                       'to_doc_level', 'to_list']
    for ag in attributes_gold:
        assert ag in dir(qa_candidate)


def test_id(span_inference_result):
    assert span_inference_result.id == "best_id_ever"


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
