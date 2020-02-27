import pytest

from farm.infer import Inferencer


@pytest.fixture(scope="module")
def adaptive_model_qa():
    model = Inferencer.load("deepset/bert-base-cased-squad2", task_type="question_answering", batch_size=16)
    return model
