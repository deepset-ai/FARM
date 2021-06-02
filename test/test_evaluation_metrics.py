import pytest
import math
from farm.evaluation.metrics import compute_metrics


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
