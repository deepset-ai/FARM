import pytest
import torch
from farm.infer import Inferencer

def test_model_versioning(caplog=None):
    # We want this load attempt to fail because we specify an invalid revision
    failed_load = None
    try:
        failed_load = Inferencer.load("deepset/roberta-base-squad2", revision="xxx", task_type="question_answering")
    except:
        pass
    assert not failed_load

    model_v1 = Inferencer.load("deepset/roberta-base-squad2", revision="v1.0", task_type="question_answering")
    model_v2 = Inferencer.load("deepset/roberta-base-squad2", revision="v2.0", task_type="question_answering")
    model_default = Inferencer.load("deepset/roberta-base-squad2", task_type="question_answering")

    weights_v1 = model_v1.model.language_model.model.encoder.layer[0].intermediate.dense.weight
    weights_v2 = model_v2.model.language_model.model.encoder.layer[0].intermediate.dense.weight
    weights_default = model_default.model.language_model.model.encoder.layer[0].intermediate.dense.weight

    assert torch.equal(weights_default, weights_v2)
    assert not torch.equal(weights_v1, weights_v2)
