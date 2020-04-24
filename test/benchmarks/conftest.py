from pathlib import Path

import pytest

from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel


@pytest.fixture(scope="session")
def onnx_adaptive_model_qa(use_gpu, num_processes):
    model_name_or_path = "deepset/bert-base-cased-squad2"
    onnx_model_export_path = Path("benchmarks/onnx-export")
    if not (onnx_model_export_path / "model.onnx").is_file():
        model = AdaptiveModel.convert_from_transformers(
            model_name_or_path, device="cpu", task_type="question_answering"
        )
        model.convert_to_onnx(onnx_model_export_path)

    model = Inferencer.load(
        onnx_model_export_path, task_type="question_answering", batch_size=1, num_processes=num_processes, gpu=use_gpu
    )

    return model
