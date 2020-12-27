import psutil
import pytest
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.infer import Inferencer, QAInferencer


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


def pytest_collection_modifyitems(items):
    for item in items:
        if "conversion" in item.nodeid:
            item.add_marker(pytest.mark.conversion)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def bert_base_squad2(request):
    model = QAInferencer.load(
            "deepset/minilm-uncased-squad2",
            task_type="question_answering",
            batch_size=4,
            num_processes=0,
            multithreading_rust=False,
            use_fast=True # TODO parametrize this to test slow as well
    )
    return model

# TODO add other model types (roberta, xlm-r, albert) here as well

@pytest.fixture(scope="module")
def distilbert_squad(request):
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    batch_size = 2
    n_epochs = 1
    evaluate_every = 4
    base_LM_model = "distilbert-base-uncased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=base_LM_model,
        do_lower_case=True,
        use_fast=True # TODO parametrize this to test slow as well
    )
    label_list = ["start_token", "end_token"]
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=20,
        doc_stride=10,
        max_query_length=6,
        train_filename="train-sample.json",
        dev_filename="dev-sample.json",
        test_filename=None,
        data_dir=Path("samples/qa"),
        label_list=label_list,
        metric="squad"
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)
    language_model = LanguageModel.load(base_LM_model)
    prediction_head = QuestionAnsweringHead()
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device
    )
    trainer.train()

    return model, processor

