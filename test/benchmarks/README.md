# Inference Speed Benchmarks

FARM provides an automated speed benchmarking pipeline with options to parameterize the benchmarks with batch_size, 
max sequence length, document size, and so on.

The pipeline is implemented using [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark). The warmup/iterations for each benchmark are configurable and the
results can be exported to a JSON file.

 

## Question Answering

The `benchmarks/question_answering.py` file contains tests for inference with PyTorch(`test_question_answering_pytorch`) 
and ONNXRuntime(`test_question_answering_onnx`).

The benchmarks are available [here](https://docs.google.com/spreadsheets/d/1ak9Cxj1zcNBDtjf7qn2j_ydKDDzpBgWiyJ7cO-7BPvA/edit?usp=sharing).

### Running Benchmark with Docker

#### GPU
For running benchmark on a GPU, bash into the Docker Image using ```docker run -it --gpus all deepset/farm-onnxruntime-gpu:0.4.3 bash```.
Once inside the container, execute ```cd FARM/test && pytest benchmarks/question_answering.py -k test_question_answering_pytorch --use_gpu --benchmark-json result.json```.

#### CPU 
Bash into the Docker container with ```docker run -it deepset/farm-inference-api:0.4.3 bash``` and then execute
 ```cd test && pytest benchmarks/question_answering.py -k test_question_answering_pytorch --benchmark-json result.json```.

### Exporting results in CSV format

The results of benchmarks are exported to a `result.json` file in the `test` folder. To convert results to csv format, 
execute `python benchmarks/convert_result_to_csv.py`.