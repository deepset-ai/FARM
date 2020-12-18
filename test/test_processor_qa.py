import logging
import json

from farm.data_handler.processor import SquadProcessor
from farm.modeling.tokenization import Tokenizer


# TODO write later
# def test_dataset_from_dicts_qa(caplog=None):
#     if caplog:
#         caplog.set_level(logging.CRITICAL)
#     sample_types = ["span", "no_answer"]
#     models = ["deepset/roberta-base-squad2"]
#     for model in models:
#         tokenizer = Tokenizer.load(pretrained_model_name_or_path=model, use_fast=False)
#         processor = SquadProcessor(tokenizer, max_seq_len=256, data_dir=None)
#     for sample_type in sample_types:
#         # clear_text = json.load(open(f"samples/qa/{sample_type}/clear_text.json"))
#         dicts = processor.file_to_dicts(f"samples/qa/{sample_type}/clear_text.json")
#         tokenized = json.load(open(f"samples/qa/{sample_type}/tokenized.json"))
#         _, _, _, baskets = processor.dataset_from_dicts(dicts, return_baskets=True)
#         print()
#
#
# if(__name__=="__main__"):
#     test_dataset_from_dicts_qa()