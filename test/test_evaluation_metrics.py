import pytest
import math
import numpy as np

from farm.evaluation.metrics import compute_metrics
from farm.evaluation.semantic_answer_similarity_evaluation import semantic_answer_similarity

def test_compute_metrics_basic():
    # check we get some exception, may not always be the AssertionError we get now
    with pytest.raises(Exception):
        compute_metrics("acc", ["x"] * 10, [""] * 11)
    ret = compute_metrics("acc", [], [])
    assert isinstance(ret, dict)
    assert "acc" in ret
    assert math.isnan(ret["acc"])
    with pytest.raises(Exception):
        compute_metrics("asdfasdf", ["a"], ["b"])
    ls = (["a"] * 5)
    ls.extend(["b"] * 5)
    ps = ["a"] * 10
    ret = compute_metrics("acc", ps, ls)
    assert ret["acc"] == 0.5
    ret = compute_metrics("acc", ls, ps)
    assert ret["acc"] == 0.5
    ret = compute_metrics("f1_macro", ps, ls)
    assert ret["f1_macro"] == 1/3
    ret = compute_metrics("f1_macro", ls, ps)
    assert ret["f1_macro"] == 1 / 3
    ret = compute_metrics(["f1_macro", "acc"], ps, ls)
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert "acc" in ret
    assert "f1_macro" in ret
    assert ret["f1_macro"] == 1/3
    assert ret["acc"] == 0.5
    ret = compute_metrics(["f1_macro", "acc", "acc"], ps, ls)
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert "acc" in ret
    assert "f1_macro" in ret
    assert ret["f1_macro"] == 1/3
    assert ret["acc"] == 0.5
    ret = compute_metrics(["f1_macro", ["acc"]], ps, ls)
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert "acc" in ret
    assert "f1_macro" in ret
    assert ret["f1_macro"] == 1/3
    assert ret["acc"] == 0.5

def test_semantic_answer_similarity(bert_base_squad2):
    bert_base_squad2.model.prediction_heads[0].n_best = 2
    result = bert_base_squad2.inference_from_file(file="samples/qa/eval-sample.json",return_json=False)

    top1_sim, topn_sim, r, d = semantic_answer_similarity(result=result,
                                                          sts_model_path_or_string="paraphrase-MiniLM-L6-v2",
                                                          debug=True)

    assert np.isclose(top1_sim, 0.7405298)
    assert np.isclose(topn_sim, 0.7405298)
    assert len(d) == 1
    assert "semantic_answer_score" in r[0].prediction[0].meta

