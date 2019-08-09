"""
Contains functions that turn readable clear text input into dictionaries of features
"""


import logging
import collections
from dotmap import DotMap

from farm.data_handler.samples import Sample
from farm.data_handler.utils import (
    truncate_seq_pair,
    expand_labels,
    add_cls_sep,
    pad,
    mask_random_words,
)

logger = logging.getLogger(__name__)


def sample_to_features_text(
    sample, label_list, max_seq_len, tokenizer, target="classification"
):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :type sample: Sample
    :param label_list: A list of all unique labels
    :type label_list: list
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :param target: Choose from "classification" and "regression"
    :type target: str
    :return: A dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    :rtype: dict
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    # tokens = tokenizer.tokenize(sample.clear_text["text"])
    tokens = sample.tokenized["tokens"]
    # tokens = sample.tokenized["word_pieces"]
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[: (max_seq_len - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    padding_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    # For inference mode
    try:
        if target == "classification":
            label_ids = label_map[sample.clear_text["label"]]
        elif target == "regression":
            label_ids = float(sample.clear_text["label"])
        else:
            # TODO Add multilabel here
            raise KeyError(target)
    except KeyError:
        label_ids = None

    feat_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
    }

    if label_ids is not None:
        feat_dict["label_ids"] = label_ids
    return [feat_dict]


def samples_to_features_ner(
    sample,
    label_list,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    pad_token="[PAD]",
    sep_token="[SEP]",
    non_initial_token="X",
    **kwargs
):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by an NER model.

    :param sample: Sample object that contains human readable text and label fields from a single NER data sample
    :type sample: Sample
    :param label_list: A list of all unique labels
    :type label_list: list
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :param cls_token: Token used to represent the beginning of the sequence
    :type cls_token: str
    :param pad_token: Token used to represent sequence padding
    :type pad_token: str
    :param sep_token: Token used to represent the border between two sequences
    :type sep_token: str
    :param non_initial_token: Token that is inserted into the label sequence in positions where there is a
                              non-word-initial token. This is done since the default NER performs prediction
                              only on word initial tokens
    :return: A dictionary containing the keys "input_ids", "padding_mask", "segment_ids", "initial_mask"
             (also "label_ids" if not in inference mode). The values are lists containing those features.
    :rtype: dict
    """

    # Tokenize words and extend the labels so they are aligned with the tokens
    # words = sample.clear_text["text"].split(" ")
    # tokens, initial_mask = words_to_tokens(words, tokenizer, max_seq_len)

    tokens = sample.tokenized["tokens"]
    initial_mask = [int(x) for x in sample.tokenized["start_of_word"]]

    # initial_mask =
    # Add CLS and SEP tokens
    tokens = add_cls_sep(tokens, cls_token, sep_token)
    initial_mask = [0] + initial_mask + [0]  # CLS and SEP don't count as initial tokens
    padding_mask = [1] * len(tokens)

    # Convert to input and labels to ids, generate masks
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if "label" in sample.clear_text:
        labels_word = sample.clear_text["label"]
        labels_token = expand_labels(labels_word, initial_mask, non_initial_token)
        # labels_token = add_cls_sep(labels_token, cls_token, sep_token)
        label_ids = [label_list.index(lt) for lt in labels_token]
    # Inference mode
    else:
        label_ids = None
    segment_ids = [0] * max_seq_len

    # Pad
    input_ids = pad(input_ids, max_seq_len, 0)
    if label_ids:
        label_ids = pad(label_ids, max_seq_len, 0)
    initial_mask = pad(initial_mask, max_seq_len, 0)
    padding_mask = pad(padding_mask, max_seq_len, 0)

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "initial_mask": initial_mask,
    }

    if label_ids:
        feature_dict["label_ids"] = label_ids

    return [feature_dict]


def samples_to_features_bert_lm(sample, max_seq_len, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :param max_seq_len: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = sample.tokenized["text_a"]["tokens"]
    tokens_b = sample.tokenized["text_b"]["tokens"]
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)

    tokens_a, t1_label = mask_random_words(tokens_a, tokenizer.vocab,
                                           token_groups=sample.tokenized["text_a"]["start_of_word"])
    tokens_b, t2_label = mask_random_words(tokens_b, tokenizer.vocab,
                                           token_groups=sample.tokenized["text_b"]["start_of_word"])
    # convert lm labels to ids
    t1_label_ids = [-1 if tok == '' else tokenizer.vocab[tok] for tok in t1_label]
    t2_label_ids = [-1 if tok == '' else tokenizer.vocab[tok] for tok in t2_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = [-1] + t1_label_ids + [-1] + t2_label_ids + [-1]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
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
        lm_label_ids.append(-1)

    # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
    if sample.clear_text["is_next_label"]:
        is_next_label_id = [0]
    else:
        is_next_label_id = [1]

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "lm_label_ids": lm_label_ids,
        "label_ids": is_next_label_id,
    }

    return [feature_dict]


def sample_to_features_squad(
    sample, tokenizer, max_seq_len, doc_stride, max_query_length
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
            # we throw it out, since there is nothing to predict.
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
