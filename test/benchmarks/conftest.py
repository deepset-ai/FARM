from pathlib import Path

import pytest

from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel


@pytest.fixture(scope="session")
def onnx_adaptive_model_qa(use_gpu, num_processes, model_name_or_path="deepset/bert-base-cased-squad2"):
    if (Path(model_name_or_path) / "model.onnx").is_file():  # load model directly if in ONNX format
        onnx_model_path = model_name_or_path
    else:  # convert to ONNX format
        onnx_model_path = Path("benchmarks/onnx-export")
        model = AdaptiveModel.convert_from_transformers(
            model_name_or_path, device="cpu", task_type="question_answering"
        )
        model.convert_to_onnx(onnx_model_path)

    try:
        model = Inferencer.load(
            onnx_model_path, task_type="question_answering", batch_size=1, num_processes=num_processes, gpu=use_gpu
        )
        yield model
    finally:
        if num_processes != 0:
            model.close_multiprocessing_pool()

