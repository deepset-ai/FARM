"""
Contains functions that turn readable clear text input into dictionaries of features
"""


import logging

from farm.data_handler.samples import Sample
from farm.data_handler.utils import (
    expand_labels,
    pad,
    mask_random_words)
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


def samples_to_features_ner(
    sample,
    tasks,
    max_seq_len,
    tokenizer,
    non_initial_token="X",
    **kwargs
):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by an NER model.

    :param sample: Sample object that contains human readable text and label fields from a single NER data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :param non_initial_token: Token that is inserted into the label sequence in positions where there is a
                              non-word-initial token. This is done since the default NER performs prediction
                              only on word initial tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask", "segment_ids", "initial_mask"
             (also "label_ids" if not in inference mode). The values are lists containing those features.
    :rtype: list
    """

    input_ids = sample.tokenized.ids
    segment_ids = sample.tokenized.type_ids

    # We construct a mask to identify the first token of a word. We will later only use them for predicting entities.
    # Special tokens don't count as initial tokens => we add 0 at the positions of special tokens
    # For BERT we add a 0 in the start and end (for CLS and SEP)
    initial_mask = _get_start_of_word(sample.tokenized.words)
    assert len(initial_mask) == len(input_ids)

    for task_name, task in tasks.items():
        try:
            label_list = task["label_list"]
            label_name = task["label_name"]
            label_tensor_name = task["label_tensor_name"]
            labels_word = sample.clear_text[label_name]
            labels_token = expand_labels(labels_word, initial_mask, non_initial_token)
            # labels_token = add_cls_sep(labels_token, cls_token, sep_token)
            label_ids = [label_list.index(lt) for lt in labels_token]
        except ValueError:
            # Usually triggered if label is not in label list
            label_ids = None
            problematic_labels = set(labels_token).difference(set(label_list))
            logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                           f"\nWe found a problem with labels {str(problematic_labels)}")
        #TODO change this when inference flag is implemented
        except KeyError:
            # Usually triggered if there is no label in the sample
            # This is expected during inference since there are no labels
            # During training, this is a problem
            label_ids = None
            logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                           "\nIf your are running in *inference* mode: Don't worry!"
                           "\nIf you are running in *training* mode: Verify you are supplying a proper label list to your processor and check that labels in input data are correct.")

        # This mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [int(x == 0) for x in sample.tokenized.attention_mask]

        feature_dict = {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
            "initial_mask": initial_mask,
        }

        if label_ids:
            feature_dict[label_tensor_name] = label_ids

    return [feature_dict]


def samples_to_features_bert_lm(sample, max_seq_len, tokenizer, next_sent_pred=True, masked_lm_prob=0.15):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :type sample: Sample
    :param max_seq_len: Maximum length of sequence.
    :type max_seq_len: int
    :param tokenizer: Tokenizer
    :param masked_lm_prob: probability of masking a token
    :type masked_lm_prob: float
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    if next_sent_pred:
        tokens_a = sample.tokenized["text_a"]["tokens"]
        tokens_b = sample.tokenized["text_b"]["tokens"]

        # mask random words
        tokens_a, t1_label = mask_random_words(tokens_a, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_a"]["start_of_word"], masked_lm_prob=masked_lm_prob)

        tokens_b, t2_label = mask_random_words(tokens_b, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_b"]["start_of_word"], masked_lm_prob=masked_lm_prob)

        # convert lm labels to ids
        t1_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t1_label]
        t2_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t2_label]
        lm_label_ids = t1_label_ids + t2_label_ids

        # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
        if sample.clear_text["nextsentence_label"]:
            is_next_label_id = [0]
        else:
            is_next_label_id = [1]
    else:
        tokens_a = sample.tokenized["text_a"]["tokens"]
        tokens_b = None
        tokens_a, t1_label = mask_random_words(tokens_a, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_a"]["start_of_word"], masked_lm_prob=masked_lm_prob)

        # convert lm labels to ids
        lm_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t1_label]

    if tokenizer.is_fast:
        seq_a_input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
        seq_b_input_ids = tokenizer.convert_tokens_to_ids(tokens_b) if next_sent_pred else None

        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=seq_a_input_ids,
                                                               token_ids_1=seq_b_input_ids)
        segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=seq_a_input_ids,
                                                                     token_ids_1=seq_b_input_ids)
        # FastTokenizers do not support get_special_tokens_mask, so special_tokens_mask is created manually
        a_token_id = tokenizer.convert_tokens_to_ids("a")
        special_tokens = tokenizer.encode_plus(text="a",
                                               text_pair="a" if next_sent_pred else None,
                                               add_special_tokens=True,
                                               return_special_tokens_mask=True,
                                               return_token_type_ids=True,
                                               return_tensors=None)
        special_tokens_grid = special_tokens["special_tokens_mask"]
        seq_a_position = special_tokens["input_ids"].index(a_token_id)
        if next_sent_pred:
            seq_b_position = special_tokens["input_ids"].index(a_token_id, seq_a_position + 1)
            special_tokens_mask = special_tokens_grid[:seq_a_position] \
                                  + [0] * len(seq_a_input_ids) \
                                  + special_tokens_grid[seq_a_position + 1:seq_b_position] \
                                  + [0] * len(seq_b_input_ids) \
                                  + special_tokens_grid[seq_b_position + 1:]
        # No next sentence prediction task, therefore no seq_b
        else:
            special_tokens_mask = special_tokens_grid[:seq_a_position] \
                                  + [0] * len(seq_a_input_ids) \
                                  + special_tokens_grid[seq_a_position + 1:]

    else:
        # encode string tokens to input_ids and add special tokens
        inputs = tokenizer.encode_plus(text=tokens_a,
                                       text_pair=tokens_b,
                                       add_special_tokens=True,
                                       truncation=False,
                                       truncation_strategy='do_not_truncate',
                                       # We've already truncated our tokens before
                                       return_special_tokens_mask=True,
                                       return_token_type_ids=True
                                       )

        input_ids = inputs["input_ids"]
        segment_ids = inputs["token_type_ids"]
        special_tokens_mask = inputs["special_tokens_mask"]

    # account for special tokens (CLS, SEP, SEP..) in lm_label_ids
    lm_label_ids = insert_at_special_tokens_pos(lm_label_ids, special_tokens_mask, insert_element=-1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4" for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)
    lm_label_ids = pad(lm_label_ids, max_seq_len, -1, pad_on_left=pad_on_left)

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "lm_label_ids": lm_label_ids,
    }

    if next_sent_pred:
        feature_dict["nextsentence_label_ids"] = is_next_label_id

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    return [feature_dict]

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
