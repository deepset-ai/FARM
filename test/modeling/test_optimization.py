import pytest

from farm.modeling.optimization import initialize_optimizer


def test_initialize_optimizer_param_schedule_opts():
    with pytest.raises(TypeError):
        initialize_optimizer(None, 1, 1, 'cpu', 0.4e-5, schedule_opts=[])
