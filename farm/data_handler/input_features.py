import logging

from farm.data_handler.utils import (
    truncate_seq_pair,
    words_to_tokens,
    expand_labels,
    add_cls_sep,
    pad,
    mask_random_words,
)

logger = logging.getLogger(__name__)


def sample_to_features_sequence(
    sample, label_list, max_seq_len, tokenizer, target="classification"
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    tokens = tokenizer.tokenize(sample.clear_text["text"])

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
    return feat_dict


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

    # Tokenize words and extend the labels so they are aligned with the tokens
    words = sample.clear_text["text"].split(" ")
    tokens, initial_mask = words_to_tokens(words, tokenizer, max_seq_len)

    labels_word = sample.clear_text["label"]
    labels_token = expand_labels(labels_word, initial_mask, non_initial_token)

    # Add CLS and SEP tokens
    tokens = add_cls_sep(tokens, cls_token, sep_token)
    labels_token = add_cls_sep(labels_token, cls_token, sep_token)
    initial_mask = [0] + initial_mask + [0]  # CLS and SEP don't count as initial tokens
    padding_mask = [1] * len(tokens)

    # Convert to input and labels to ids, generate masks
    # Todo: Something is odd here because [PAD] is index one in the vocab of tokenizer but we are padding with 0, or maybe it doesnt matter because its masked out anyways
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if labels_word:
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

    return feature_dict


def samples_to_features_bert_lm(sample, max_seq_len, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.
    :param example: Sample, containing sentence input as strings and is_next label
    :param max_seq_len: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = tokenizer.tokenize(sample.clear_text["text_a"])
    tokens_b = tokenizer.tokenize(sample.clear_text["text_b"])
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)

    tokens_a, t1_label = mask_random_words(tokens_a, tokenizer)
    tokens_b, t2_label = mask_random_words(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]

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

    return feature_dict
