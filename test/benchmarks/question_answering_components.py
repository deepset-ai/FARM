from farm.infer import Inferencer
from farm.data_handler.processor import SquadProcessor, Processor
from pprint import pprint

# Variables

modelnames = ["deepset/bert-base-cased-squad2", "deepset/minilm-uncased-squad2"]

batch_size = 32
document_size = 100000
num_processes = 1
iterations = 3
max_seq_len = 384
doc_stride = 128
gpu = True
task_type = "question_answering"
sample_file = "question_answering_sample.txt"
qs = ["When were the first traces of Human life found in France?",
      "How many pretrained models are available in Transformers?",
      "What does Transformers provide?",
      "Transformers provides interoperability between which frameworks?"]

def prepare_dict(sample_file, q):
    with open(sample_file) as f:
        text = f.read()[:document_size]
        assert len(text) == document_size
    dicts = [{"qas": [q], "context": text}]
    return dicts

def analyse_timing(lm_only, full):
    lm_only_preproc = lm_only["dataset_single_proc"] - lm_only["init"]
    full_preproc = full["dataset_single_proc"] - full["init"]
    ave_preproc = (lm_only_preproc + full_preproc) / 2
    lm_time = lm_only["formatted_preds"] - lm_only["dataset_single_proc"]

    init_to_forward_lm = lm_only["forward"] - lm_only["init"]
    init_to_formatted_full = full["formatted_preds"] - full["init"]
    ph_time = init_to_formatted_full - init_to_forward_lm

    total = full["formatted_preds"] - full["init"]
    return ave_preproc, lm_time, ph_time, total

"""
Various timing check points are set up in the Inferencer (see diagram below).
It can be run with a real ph or with a dummy ph.

To measure time of lm, we need to run up to forward without ph
To measure time of ph, full pipeline time - lm time

init            dataset         forward         formatted
  |     proc       |     ph         |       ph      |
  |                |     lm         |               |
"""

results = []
for q in qs[:2]:
    for modelname in modelnames:
        dicts = prepare_dict(sample_file, q)

        inferencer_dummy_ph = Inferencer.load(modelname,
                                     batch_size=batch_size,
                                     gpu=gpu,
                                     task_type=task_type,
                                     max_seq_len=max_seq_len,
                                     num_processes=num_processes,
                                     doc_stride=doc_stride,
                                     benchmark_lm=True)
        inferencer_dummy_ph.inference_from_dicts(dicts)
        lm_only_timing = inferencer_dummy_ph.timing

        inferencer_real_ph = Inferencer.load(modelname,
                                     batch_size=batch_size,
                                     gpu=gpu,
                                     task_type=task_type,
                                     max_seq_len=max_seq_len,
                                     num_processes=num_processes,
                                     doc_stride=doc_stride,
                                     benchmark_lm=False)
        inferencer_real_ph.inference_from_dicts(dicts)
        full_timing = inferencer_real_ph.timing

        ave_preproc, lm_time, ph_time, total = analyse_timing(lm_only_timing, full_timing)
        result = {"model name": modelname,
                  "question": q,
                  "preproc": ave_preproc,
                  "language model": lm_time,
                  "prediction head": ph_time,
                  "total": total}
        results.append(result)

for result in results:
    pprint(result)
    print()