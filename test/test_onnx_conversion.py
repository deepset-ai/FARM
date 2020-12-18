import numpy as np
import pytest

from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel


@pytest.mark.parametrize("model_name", ["deepset/bert-base-cased-squad2", "deepset/roberta-base-squad2"])
def test_onnx_conversion_and_inference(tmp_path, model_name):
    AdaptiveModel.convert_to_onnx(
        model_name=model_name, output_path=tmp_path / "test-onnx", task_type="question_answering"
    )
    onnx_inferencer = Inferencer.load(tmp_path / "test-onnx", task_type="question_answering", num_processes=0)
    qa_input = [
        {
            "questions": ["What is the population of Berlin?"],
            "text": "Berlin is the capital and largest city of Germany by both area and population. Its 3,769,495 "
            "inhabitants as of December 31, 2019 make it the most populous city of the European Union, "
            "according to population within city limits.The city is also one of Germany's 16 federal states.",
        }
    ]
    result_onnx = onnx_inferencer.inference_from_dicts(qa_input)[0]
    assert result_onnx["predictions"][0]["answers"][0]["answer"] == "3,769,495"

    pytorch_inferencer = Inferencer.load(model_name, task_type="question_answering", num_processes=0)
    result_pytorch = pytorch_inferencer.inference_from_dicts(qa_input)[0]

    for (onnx, pytorch) in zip(
        result_onnx["predictions"][0]["answers"][0].items(), result_pytorch["predictions"][0]["answers"][0].items()
    ):
        # keys
        assert onnx[0] == pytorch[0]
        # values
        if type(onnx[1]) == float:
            np.testing.assert_almost_equal(onnx[1], pytorch[1], decimal=4)  # score
        else:
            assert onnx[1] == pytorch[1]
