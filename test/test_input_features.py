import json
import logging

from farm.data_handler.input_features import sample_to_features_qa
from farm.data_handler.samples import Sample
from farm.modeling.tokenization import Tokenizer


MODEL = "roberta-base"
SP_TOKENS_START = 1
SP_TOKENS_MID = 2

def to_list(x):
    try:
        return x.tolist()
    except:
        return x

def test_sample_to_features_qa(caplog):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    sample_types = ["span", "no_answer"]

    for sample_type in sample_types:
        clear_text = json.load(open(f"samples/qa/{sample_type}/clear_text.json"))
        tokenized = json.load(open(f"samples/qa/{sample_type}/tokenized.json"))
        features_gold = json.load(open(f"samples/qa/{sample_type}/features.json"))
        max_seq_len = len(features_gold["input_ids"])

        tokenizer = Tokenizer.load(pretrained_model_name_or_path=MODEL, do_lower_case=False)
        curr_id = "-".join([str(x) for x in features_gold["id"]])

        s = Sample(id=curr_id, clear_text=clear_text, tokenized=tokenized)
        features = sample_to_features_qa(s, tokenizer, max_seq_len, SP_TOKENS_START, SP_TOKENS_MID)[0]
        features = to_list(features)

        keys = features_gold.keys()
        for k in keys:
            value_gold = features_gold[k]
            value = to_list(features[k])
            assert value == value_gold, f"Mismatch between the {k} features in the {sample_type} test sample."

if __name__ == "__main__":
    test_sample_to_features_qa(None)
