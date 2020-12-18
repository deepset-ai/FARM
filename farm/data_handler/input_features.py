"""
Contains functions that turn readable clear text input into dictionaries of features
"""


import logging

from farm.data_handler.samples import Sample
from farm.data_handler.utils import (
    expand_labels,
    pad)
from farm.modeling.tokenization import insert_at_special_tokens_pos

import numpy as np

logger = logging.getLogger(__name__)


def sample_to_features_text(
    sample, tasks, max_seq_len, tokenizer
):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    :rtype: list
    """

    if tokenizer.is_fast:
        text = sample.clear_text["text"]
        # Here, we tokenize the sample for the second time to get all relevant ids
        # This should change once we git rid of FARM's tokenize_with_metadata()
        inputs = tokenizer(text,
                           return_token_type_ids=True,
                           truncation=True,
                           truncation_strategy="longest_first",
                           max_length=max_seq_len,
                           return_special_tokens_mask=True)

        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != len(sample.tokenized["tokens"]):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to "
                         f"{len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs "
                         f"from number of tokens produced in tokenize_with_metadata(). \n"
                         f"Further processing is likely to be wrong.")
    else:
        # TODO It might be cleaner to adjust the data structure in sample.tokenized
        tokens_a = sample.tokenized["tokens"]
        tokens_b = sample.tokenized.get("tokens_b", None)

        inputs = tokenizer.encode_plus(
            tokens_a,
            tokens_b,
            add_special_tokens=True,
            truncation=False,  # truncation_strategy is deprecated
            return_token_type_ids=True,
            is_split_into_words=False,
        )

    input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feat_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
    }

    # Add Labels for different tasks
    for task_name, task in tasks.items():
        try:
            label_name = task["label_name"]
            label_raw = sample.clear_text[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                try:
                    label_ids = [label_list.index(label_raw)]
                except ValueError as e:
                    raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            elif task["task_type"] == "regression":
                label_ids = [float(label_raw)]
            else:
                raise ValueError(task["task_type"])
        except KeyError:
            # For inference mode we don't expect labels
            label_ids = None
        if label_ids is not None:
            feat_dict[task["label_tensor_name"]] = label_ids
    return [feat_dict]


#TODO remove once NQ processing is adjusted
def get_roberta_seq_2_start(input_ids):
    # This commit (https://github.com/huggingface/transformers/commit/dfe012ad9d6b6f0c9d30bc508b9f1e4c42280c07)from
    # huggingface transformers now means that RobertaTokenizer.encode_plus returns only zeros in token_type_ids. Therefore, we need
    # another way to infer the start of the second input sequence in RoBERTa. Roberta input sequences have the following
    # format: <s> P1 </s> </s> P2 </s>
    # <s> has index 0 and </s> has index 2. To find the beginning of the second sequence, this function first finds
    # the index of the second </s>
    first_backslash_s = input_ids.index(2)
    second_backslash_s = input_ids.index(2, first_backslash_s + 1)
    return second_backslash_s + 1

#TODO remove once NQ processing is adjusted
def get_camembert_seq_2_start(input_ids):
    # CamembertTokenizer.encode_plus returns only zeros in token_type_ids (same as RobertaTokenizer).
    # This is another way to find the start of the second sequence (following get_roberta_seq_2_start)
    # Camembert input sequences have the following
    # format: <s> P1 </s> </s> P2 </s>
    # <s> has index 5 and </s> has index 6. To find the beginning of the second sequence, this function first finds
    # the index of the second </s>
    first_backslash_s = input_ids.index(6)
    second_backslash_s = input_ids.index(6, first_backslash_s + 1)
    return second_backslash_s + 1


# def _SQUAD_improve_answer_span(
#     doc_tokens, input_start, input_end, tokenizer, orig_answer_text
# ):
#     """Returns tokenized answer spans that better match the annotated answer."""
#
#     # The SQuAD annotations are character based. We first project them to
#     # whitespace-tokenized words. But then after WordPiece tokenization, we can
#     # often find a "better match". For example:
#     #
#     #   Question: What year was John Smith born?
#     #   Context: The leader was John Smith (1895-1943).
#     #   Answer: 1895
#     #
#     # The original whitespace-tokenized answer will be "(1895-1943).". However
#     # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
#     # the exact answer, 1895.
#     #
#     # However, this is not always possible. Consider the following:
#     #
#     #   Question: What country is the top exporter of electornics?
#     #   Context: The Japanese electronics industry is the lagest in the world.
#     #   Answer: Japan
#     #
#     # In this case, the annotator chose "Japan" as a character sub-span of
#     # the word "Japanese". Since our WordPiece tokenizer does not split
#     # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
#     # in SQuAD, but does happen.
#     tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
#
#     for new_start in range(input_start, input_end + 1):
#         for new_end in range(input_end, new_start - 1, -1):
#             text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
#             if text_span == tok_answer_text:
#                 return (new_start, new_end)
#
#     return (input_start, input_end)
