import pytest

from farm.infer import Inferencer


@pytest.fixture(scope="session")
def adaptive_model_qa(use_gpu, num_processes):
    model = Inferencer.load(
        "deepset/bert-base-cased-squad2",
        task_type="question_answering",
        batch_size=16,
        num_processes=num_processes,
        gpu=use_gpu,
    )
    return model
