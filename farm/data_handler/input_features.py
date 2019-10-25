"""
Contains functions that turn readable clear text input into dictionaries of features
"""


import logging
import collections
from dotmap import DotMap
import numpy as np

from farm.data_handler.samples import Sample
from farm.data_handler.utils import (
    expand_labels,
    pad,
    mask_random_words,
)
from farm.modeling.tokenization import insert_at_special_tokens_pos

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

    #TODO It might be cleaner to adjust the data structure in sample.tokenized
    # Verify if this current quickfix really works for pairs
    tokens_a = sample.tokenized["tokens"]
    tokens_b = sample.tokenized.get("tokens_b", None)

    inputs = tokenizer.encode_plus(
        tokens_a,
        tokens_b,
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation_strategy='do_not_truncate' # We've already truncated our tokens before
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

    tokens = sample.tokenized["tokens"]
    inputs = tokenizer.encode_plus(text=tokens,
                                   text_pair=None,
                                   add_special_tokens=True,
                                   max_length=max_seq_len,
                                   truncation_strategy='do_not_truncate' # We've already truncated our tokens before
    )

    input_ids, segment_ids, special_tokens_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["special_tokens_mask"]

    # We construct a mask to identify the first token of a word. We will later only use them for predicting entities.
    # Special tokens don't count as initial tokens => we add 0 at the positions of special tokens
    # For BERT we add a 0 in the start and end (for CLS and SEP)
    initial_mask = [int(x) for x in sample.tokenized["start_of_word"]]
    initial_mask = insert_at_special_tokens_pos(initial_mask, special_tokens_mask, insert_element=0)
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
            label_ids = None
            problematic_labels = set(labels_token).difference(set(label_list))
            logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                           f"\nWe found a problem with labels {str(problematic_labels)}")
        except KeyError:
            # For inference mode we don't expect labels
            label_ids = None
            logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                           "\nIf your are running in *inference* mode: Don't worry!"
                           "\nIf you are running in *training* mode: Verify you are supplying a proper label list to your processor and check that labels in input data are correct.")

        # This mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)

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
        initial_mask = pad(initial_mask, max_seq_len, 0, pad_on_left=pad_on_left)
        if label_ids:
            label_ids = pad(label_ids, max_seq_len, 0, pad_on_left=pad_on_left)

        feature_dict = {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
            "initial_mask": initial_mask,
        }

        if label_ids:
            feature_dict[label_tensor_name] = label_ids

    return [feature_dict]


def samples_to_features_bert_lm(sample, max_seq_len, tokenizer, next_sent_pred=True):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :type sample: Sample
    :param max_seq_len: Maximum length of sequence.
    :type max_seq_len: int
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    if next_sent_pred:
        tokens_a = sample.tokenized["text_a"]["tokens"]
        tokens_b = sample.tokenized["text_b"]["tokens"]

        # mask random words
        tokens_a, t1_label = mask_random_words(tokens_a, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_a"]["start_of_word"])

        tokens_b, t2_label = mask_random_words(tokens_b, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_b"]["start_of_word"])
        # convert lm labels to ids
        t1_label_ids = [-1 if tok == '' else tokenizer.vocab[tok] for tok in t1_label]
        t2_label_ids = [-1 if tok == '' else tokenizer.vocab[tok] for tok in t2_label]
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
                                               token_groups=sample.tokenized["text_a"]["start_of_word"])
        # convert lm labels to ids
        lm_label_ids = [-1 if tok == '' else tokenizer.vocab[tok] for tok in t1_label]

    # encode string tokens to input_ids and add special tokens
    inputs = tokenizer.encode_plus(text=tokens_a,
                                   text_pair=tokens_b,
                                   add_special_tokens=True,
                                   max_length=max_seq_len,
                                   truncation_strategy='do_not_truncate'
                                   # We've already truncated our tokens before
                                   )

    input_ids, segment_ids, special_tokens_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
        "special_tokens_mask"]

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


def sample_to_features_squad(
    sample, tokenizer, max_seq_len, doc_stride, max_query_length, tasks,
):
    sample.clear_text = DotMap(sample.clear_text, _dynamic=False)
    is_training = sample.clear_text.is_training

    unique_id = 1000000000
    features = []

    query_tokens = tokenizer.tokenize(sample.clear_text.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(sample.clear_text.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and sample.clear_text.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not sample.clear_text.is_impossible:
        tok_start_position = orig_to_tok_index[sample.clear_text.start_position]
        if sample.clear_text.end_position < len(sample.clear_text.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[sample.clear_text.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _SQUAD_improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            tokenizer,
            sample.clear_text.orig_answer_text,
        )

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_len - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"]
    )
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _SQUAD_check_is_max_context(
                doc_spans, doc_span_index, split_token_index
            )
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            padding_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(padding_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        start_position = 0
        end_position = 0
        if is_training and not sample.clear_text.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we keep it but set the start and end position to unanswerable
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if is_training and sample.clear_text.is_impossible:
            start_position = 0
            end_position = 0

        inp_feat = {}
        inp_feat["input_ids"] = input_ids
        inp_feat["padding_mask"] = padding_mask  # attention_mask
        inp_feat["segment_ids"] = segment_ids  # token_type_ids
        inp_feat["start_position"] = start_position
        inp_feat["end_position"] = end_position
        inp_feat["is_impossible"] = sample.clear_text.is_impossible
        features.append(inp_feat)
        unique_id += 1

    return features


def _SQUAD_improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _SQUAD_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
