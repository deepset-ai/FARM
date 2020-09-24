import pytest

from farm.modeling.adaptive_model import AdaptiveModel
from farm.infer import Inferencer


@pytest.mark.parametrize("model_name", ["deepset/bert-base-cased-squad2", "deepset/roberta-base-squad2"])
def test_onnx_conversion_roberta(tmp_path, model_name):
    AdaptiveModel.convert_to_onnx(model_name=model_name, output_path=tmp_path/"test-onnx", task_type="question_answering")
    inferencer = Inferencer.load(tmp_path/"test-onnx", task_type="question_answering")
    qa_input = [
        {
            "qas": ["What is the population of Berlin?"],
            "context": "Berlin is the capital and largest city of Germany by both area and population. Its 3,769,495 "
                       "inhabitants as of December 31, 2019 make it the most populous city of the European Union, "
                       "according to population within city limits.The city is also one of Germany's 16 federal states."
        }
    ]
    results = inferencer.inference_from_dicts(qa_input)
    assert results[0]["predictions"][0]["answers"][0]["answer"] == "3,769,495"
