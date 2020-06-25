import psutil
import pytest

from farm.infer import Inferencer


def pytest_addoption(parser):
    """
    Hook to pass pytest-fixture arguments to tests from the command line.
    """
    parser.addoption("--use_gpu", action="store_true", default=False)


def pytest_generate_tests(metafunc):
    """
    This method gets called for all test cases. Here, we set the arguments supplied in pytest_addoption().
    """
    option_value = metafunc.config.option.use_gpu
    if 'use_gpu' in metafunc.fixturenames:
        if option_value:
            metafunc.parametrize("use_gpu", [True], scope="session")
        else:
            metafunc.parametrize("use_gpu", [False], scope="session")


@pytest.fixture()
def adaptive_model_qa(use_gpu, num_processes):
    """
    PyTest Fixture for a Question Answering Inferencer based on PyTorch.
    """
    try:
        model = Inferencer.load(
            "deepset/bert-base-cased-squad2",
            task_type="question_answering",
            batch_size=16,
            num_processes=num_processes,
            gpu=use_gpu,
        )
        yield model
    finally:
        if num_processes != 0:
            # close the pool
            # we pass join=True to wait for all sub processes to close
            # this is because below we want to test if all sub-processes
            # have exited
            model.close_multiprocessing_pool(join=True)

    # check if all workers (sub processes) are closed
    current_process = psutil.Process()
    children = current_process.children()
    assert len(children) == 0
