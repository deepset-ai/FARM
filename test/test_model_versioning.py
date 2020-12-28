import pytest
import torch
from farm.infer import Inferencer

def test_wrong_revision(caplog=None):
    # We want this load attempt to fail because we specify an invalid revision
    failed_load = None
    try:
        failed_load = Inferencer.load("deepset/roberta-base-squad2", revision="xxx", task_type="question_answering")
    except:
        pass
    assert not failed_load

def test_revision_v1(caplog=None):
    model = Inferencer.load("deepset/roberta-base-squad2", revision="v1.0", task_type="question_answering")
    assert torch.isclose(torch.sum(model.model.language_model.model.encoder.layer[0].intermediate.dense.weight),
                torch.sum(torch.tensor([-21394.6055])))
    del model

def test_revision_v2(caplog=None):
    model = Inferencer.load("deepset/roberta-base-squad2", revision="v2.0", task_type="question_answering")
    assert torch.isclose(torch.sum(model.model.language_model.model.encoder.layer[0].intermediate.dense.weight),
                       torch.sum(torch.tensor([-21411.4414])))
    del model

def test_revision_default(caplog=None):
    # default model should be the same as v2
    model = Inferencer.load("deepset/roberta-base-squad2", task_type="question_answering")
    assert torch.isclose(
        torch.sum(model.model.language_model.model.encoder.layer[0].intermediate.dense.weight),
        torch.sum(torch.tensor([-21411.4414])))
    del model
