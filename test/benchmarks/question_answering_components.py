"""
This benchmarks the time taken by preprocessing / language modelling / prediction head processing.
This is done by running the Inferencer twice: once with ph enabled and once with ph disabled.
The Inferencer contains a Benchmarker object which measures the time taken by preprocessing and model processing.
"""

from farm.infer import Inferencer
from pprint import pformat
import pandas as pd
from tqdm import tqdm
import logging
import json
from datetime import date


logger = logging.getLogger(__name__)

task_type = "question_answering"
sample_file = "samples/question_answering_sample.txt"
questions_file = "samples/question_answering_questions.txt"
num_processes = 1
passages_per_char = 2400 / 1000000      # numerator is number of passages when 1mill chars paired with one of the questions, msl 384, doc stride 128
# date_str = date.today().strftime("%d_%m_%Y")
output_file = f"results_per_component.csv"

params = {
    "modelname": ["deepset/bert-base-cased-squad2", "deepset/minilm-uncased-squad2", "deepset/roberta-base-squad2", "deepset/bert-large-uncased-whole-word-masking-squad2", "deepset/xlm-roberta-large-squad2"],
    "batch_size": [50],
    "document_size": [1000_000],
    "max_seq_len": [384],       # This param needs to be set to 384 otherwise the "passages per sec" calculation will be wrong
    "doc_stride": [128],        # This param needs to be set to 128 otherwise the "passages per sec" calculation will be wrong
    "gpu": [True],
    "question": [l[:-1] for l in open(questions_file)][:2]
}

def benchmark(params, output=output_file):
    ds = generate_param_dicts(params)
    print(f"Running {len(ds)} benchmarks...")
    results = []
    warmup_run()
    for d in tqdm(ds):
        result = benchmark_single(**d)
        results.append(result)
        df = pd.DataFrame.from_records(results)
        df.to_csv(output)
        logger.info("\n\n" + pformat(result) + "\n")
        with open(output_file.replace(".csv", ".md"), "w") as f:
            f.write(str(df.to_markdown()))

def warmup_run():
    """ This run warms up the gpu. We saw cases where the first run in the loop took longer or showed different
    time profile characteristics. This warm up run is intended to reduce this kind of fluctation. """
    question = [l[:-1] for l in open(questions_file)][0]
    document_size = 100_000
    input_dict = prepare_dict(sample_file, question, document_size)
    # Run once with real prediction heads
    inferencer = Inferencer.load("deepset/bert-base-cased-squad2",
                                 batch_size=16,
                                 gpu=True,
                                 task_type=task_type,
                                 max_seq_len=384,
                                 num_processes=num_processes,
                                 doc_stride=128,
                                 dummy_ph=False,
                                 benchmarking=True)
    inferencer.inference_from_dicts(input_dict)


def benchmark_single(batch_size, gpu, max_seq_len, doc_stride, document_size, question, modelname):
    try:
        input_dict = prepare_dict(sample_file, question, document_size)

        # Run once with dummy prediction heads
        inferencer_dummy_ph = Inferencer.load(modelname,
                                              batch_size=batch_size,
                                              gpu=gpu,
                                              task_type=task_type,
                                              max_seq_len=max_seq_len,
                                              num_processes=num_processes,
                                              doc_stride=doc_stride,
                                              dummy_ph=True,
                                              benchmarking=True)
        inferencer_dummy_ph.inference_from_dicts(input_dict)
        preproc_lm_only, model_lm_only = inferencer_dummy_ph.benchmarker.summary()

        # Run once with real prediction heads
        inferencer_real_ph = Inferencer.load(modelname,
                                             batch_size=batch_size,
                                             gpu=gpu,
                                             task_type=task_type,
                                             max_seq_len=max_seq_len,
                                             num_processes=num_processes,
                                             doc_stride=doc_stride,
                                             dummy_ph=False,
                                             benchmarking=True)
        inferencer_real_ph.inference_from_dicts(input_dict)
        preproc_full, model_full = inferencer_real_ph.benchmarker.summary()

        ave_preproc, lm_time, ph_time, total = analyse_timing(preproc_lm_only, model_lm_only, preproc_full, model_full)
        result = {"model name": modelname,
                  "question": question,
                  "preproc": ave_preproc,
                  "language model": lm_time,
                  "prediction head": ph_time,
                  "total": total,
                  "passages per sec": (document_size * passages_per_char) / total,
                  "batch_size": batch_size,
                  "document_size": document_size,
                  "num_processes": num_processes,
                  "max_seq_len": max_seq_len,
                  "doc_stride": doc_stride,
                  "gpu": gpu,
                  "sample_file": sample_file,
                  "error": ""
                  }
        del inferencer_dummy_ph
        del inferencer_real_ph
    except Exception as e:
        result = {"model name": modelname,
                  "question": question,
                  "preproc": -1,
                  "language model": -1,
                  "prediction head": -1,
                  "total": -1,
                  "passages_per_sec": -1,
                  "batch_size": batch_size,
                  "document_size": document_size,
                  "num_processes": num_processes,
                  "max_seq_len": max_seq_len,
                  "doc_stride": doc_stride,
                  "gpu": gpu,
                  "sample_file": sample_file,
                  "error": str(e)
                  }
    return result


def generate_param_dicts(params):
    state = {}
    result = []
    params = {k: list(v) for k, v in params.items()}
    param_names = list(params)
    recurse(param_names, state, result)
    return result

def recurse(param_names, state, result):
    if len(param_names) == 0:
        result.append(dict(state))
        return

    curr_pn = param_names[0]
    param_names = param_names[1:]
    for x in params[curr_pn]:
        state[curr_pn] = x
        recurse(param_names, state, result)


def prepare_dict(sample_file, q, document_size):
    with open(sample_file) as f:
        if sample_file[-3:] == "txt":
            text = f.read()[:document_size]
            assert len(text) == document_size
            dicts = [{"questions": [q], "text": text}]
        elif sample_file[-4:] == "json":
            data = json.load(f)
            dicts = []
            for d in data["data"]:
                for p in d["paragraphs"]:
                    dicts.append(p)
    return dicts


def analyse_timing(preproc_lm_only, model_lm_only, preproc_full, model_full):
    ave_preproc = (preproc_lm_only + preproc_full) / 2
    lm_time = model_lm_only

    # init_to_formatted_lm = preproc_lm_only + model_lm_only
    # init_to_formatted_full = preproc_full + model_full
    # ph_time = init_to_formatted_full - init_to_formatted_lm

    ph_time = model_full - model_lm_only
    total = preproc_full + model_full

    return ave_preproc, lm_time, ph_time, total

if __name__ == "__main__":

    benchmark(params)
