import logging

import pytest
import torch

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("max_seq_len", [128, 256, 384])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("document_size", [10_000, 100_000])
@pytest.mark.parametrize("num_processes", [0], scope="session")
def test_question_answering_pytorch(adaptive_model_qa, benchmark, max_seq_len, batch_size, use_gpu, document_size):
    if use_gpu and not torch.cuda.is_available():
        pytest.skip("Skipping benchmarking on GPU as it not available.")

    if not use_gpu and document_size > 10_000:
        pytest.skip("Document size is large for CPU")

    with open("benchmarks/sample_file.txt") as f:
        context = f.read()[:document_size]
    QA_input = [{"qas": ["When were the first traces of Human life found in France?"], "context": context}]

    adaptive_model_qa.batch_size = batch_size
    adaptive_model_qa.max_seq_len = max_seq_len
    benchmark.pedantic(
        target=adaptive_model_qa.inference_from_dicts, args=(QA_input,), warmup_rounds=1, iterations=3,
    )


@pytest.mark.parametrize("max_seq_len", [128, 256, 384])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("document_size", [10_000, 100_000])
@pytest.mark.parametrize("num_processes", [0], scope="session")
def test_question_answering_onnx(onnx_adaptive_model_qa, benchmark, max_seq_len, batch_size, use_gpu, document_size):
    if use_gpu and not torch.cuda.is_available():
        pytest.skip("Skipping benchmarking on GPU as it not available.")

    if not use_gpu and document_size > 10_000:
        pytest.skip("Document size is large for CPU")

    with open("benchmarks/sample_file.txt") as f:
        context = f.read()[:document_size]
    QA_input = [{"qas": ["When were the first traces of Human life found in France?"], "context": context}]

    onnx_adaptive_model_qa.batch_size = batch_size
    onnx_adaptive_model_qa.max_seq_len = max_seq_len
    benchmark.pedantic(
        target=onnx_adaptive_model_qa.inference_from_dicts, args=(QA_input,), warmup_rounds=1, iterations=3
    )
