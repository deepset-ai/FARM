import logging

import pytest
import torch

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("framework", ["pytorch", "onnx"])
@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("max_seq_len", [128, 256, 384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_question_answering(
    adaptive_model_qa, onnx_adaptive_model_qa, benchmark, framework, max_seq_len, batch_size, use_gpu
):
    with open("benchmarks/sample_file.txt") as f:
        context = f.read()
    QA_input = [{"qas": ["When were the first traces of Human life found in France?"], "context": context}]

    if use_gpu and not torch.cuda.is_available():
        pytest.skip("Skipping benchmarking on GPU as it not available.")

    if framework == "pytorch":
        adaptive_model_qa.batch_size = batch_size
        adaptive_model_qa.max_seq_len = max_seq_len
        benchmark.pedantic(
            target=adaptive_model_qa.inference_from_dicts, args=(QA_input,), warmup_rounds=1, iterations=3,
        )

    elif framework == "onnx":
        onnx_adaptive_model_qa.batch_size = batch_size
        onnx_adaptive_model_qa.max_seq_len = max_seq_len
        benchmark.pedantic(
            target=onnx_adaptive_model_qa.inference_from_dicts, args=(QA_input,), warmup_rounds=1, iterations=3
        )
    else:
        raise Exception(f"Benchmarking for framework {framework} is not supported.")
