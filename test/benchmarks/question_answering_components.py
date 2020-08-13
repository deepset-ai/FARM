"""
This benchmarks the time taken by preprocessing / language modelling / prediction head processing.
This is done by running the Inferencer twice: once with ph enabled and once with ph disabled.
The Inferencer contains a Benchmarker object which measures the time taken by preprocessing and model processing.
"""



from farm.infer import Inferencer
from farm.data_handler.processor import SquadProcessor, Processor
from pprint import pprint
import pandas as pd

# Variables

modelnames = ["deepset/bert-base-cased-squad2", "deepset/minilm-uncased-squad2", "deepset/roberta-base-squad2", "deepset/bert-large-uncased-whole-word-masking-squad2", "deepset/xlm-roberta-large-squad2"]

batch_size = 32
document_size = 100000
num_processes = 1
iterations = 3
max_seq_len = 384
doc_stride = 128
gpu = True
task_type = "question_answering"
sample_file = "question_answering_sample.txt"
questions_file = "question_answering_questions.txt"
qs = [l for l in open(questions_file)]

def prepare_dict(sample_file, q):
    with open(sample_file) as f:
        text = f.read()[:document_size]
        assert len(text) == document_size
    dicts = [{"qas": [q], "context": text}]
    return dicts

def analyse_timing(preproc_lm_only, model_lm_only, preproc_full, model_full):
    ave_preproc = (preproc_lm_only + preproc_full) / 2
    lm_time = preproc_lm_only + model_lm_only

    init_to_formatted_lm = preproc_lm_only + model_lm_only
    init_to_formatted_full = preproc_full + model_full
    ph_time = init_to_formatted_full - init_to_formatted_lm

    total = init_to_formatted_full
    return ave_preproc, lm_time, ph_time, total

results = []
for q in qs[:2]:
    for modelname in modelnames:
        dicts = prepare_dict(sample_file, q[:-1])

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
        inferencer_dummy_ph.inference_from_dicts(dicts)
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
        inferencer_real_ph.inference_from_dicts(dicts)
        preproc_full, model_full = inferencer_real_ph.benchmarker.summary()

        ave_preproc, lm_time, ph_time, total = analyse_timing(preproc_lm_only, model_lm_only, preproc_full, model_full)
        result = {"model name": modelname,
                  "question": q[:-1],
                  "preproc": ave_preproc,
                  "language model": lm_time,
                  "prediction head": ph_time,
                  "total": total,
                  "batch_size": batch_size,
                  "document_size": document_size,
                  "num_processes": num_processes,
                  "max_seq_len": max_seq_len,
                  "doc_stride": doc_stride,
                  "gpu": gpu,
                  "sample_file": sample_file,
                  }
        results.append(result)

for result in results:
    pprint(result)
    print()

df = pd.DataFrame.from_records(results)
df.to_csv("component_test.csv")