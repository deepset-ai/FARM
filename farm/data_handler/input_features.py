import logging

from farm.data_handler.utils import (
    truncate_seq_pair,
    words_to_tokens,
    expand_labels,
    add_cls_sep,
    pad,
    print_example_with_features,
    mask_random_words,
)

logger = logging.getLogger(__name__)


# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(self, input_ids, padding_mask, segment_ids, label_id, initial_mask=None):
#         self.input_ids = input_ids
#         self.padding_mask = padding_mask
#         self.segment_ids = segment_ids
#         self.label_id = label_id
#         self.initial_mask = initial_mask
#         self.order = [
#             self.input_ids,
#             self.padding_mask,
#             self.segment_ids,
#             self.label_id,
#             self.initial_mask,
#         ]

# class InputFeature(object):
#     def __init__(self, name, value, type):
#         self.name = "padding_mask"
#         self.value = [0,0,0,0,1,1,1,1,1]
#         self.type = torch.long


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, **kwargs):
        """

        :param features: dict, key = name of feature, value = list or ndarray
        """
        self.features = kwargs


def examples_to_features_sequence(
    examples, label_list, max_seq_len, tokenizer, target="classification"
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[: (max_seq_len - 2)]

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
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

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

        if target == "classification":
            label_id = label_map[example.label]
        elif target == "regression":
            label_id = float(example.label)
        else:
            # TODO Add multilabel here
            raise KeyError(target)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("padding_mask: %s" % " ".join([str(x) for x in padding_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                padding_mask=padding_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def examples_to_features_ner(
    examples,
    label_list,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    pad_token="[PAD]",
    sep_token="[SEP]",
    non_initial_token="X",
    **kwargs
):

    feature_objects = []

    for idx, example in enumerate(examples):
        # Tokenize words and extend the labels so they are aligned with the tokens
        words = example.text_a.split(" ")
        tokens, initial_mask = words_to_tokens(words, tokenizer, max_seq_len)

        labels_word = example.label
        labels_token = expand_labels(labels_word, initial_mask, non_initial_token)

        # Add CLS and SEP tokens
        tokens = add_cls_sep(tokens, cls_token, sep_token)
        labels_token = add_cls_sep(labels_token, cls_token, sep_token)
        initial_mask = (
            [0] + initial_mask + [0]
        )  # CLS and SEP don't count as initial tokens
        padding_mask = [1] * len(tokens)

        # Convert to input and labels to ids, generate masks
        # Todo: Something is odd here because [PAD] is index one in the vocab of tokenizer but we are padding with 0, or maybe it doesnt matter because its masked out anyways
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_list.index(lt) for lt in labels_token]
        segment_ids = [0] * max_seq_len

        # Pad
        input_ids = pad(input_ids, max_seq_len, 0)
        label_ids = pad(label_ids, max_seq_len, 0)
        initial_mask = pad(initial_mask, max_seq_len, 0)
        padding_mask = pad(padding_mask, max_seq_len, 0)

        if idx < 2:
            print_example_with_features(
                example,
                tokens,
                input_ids,
                padding_mask,
                segment_ids,
                label_ids,
                initial_mask,
            )

        feature_object = InputFeatures(
            input_ids=input_ids,
            padding_mask=padding_mask,
            segment_ids=segment_ids,
            label_id=label_ids,
            initial_mask=initial_mask,
        )
        feature_objects.append(feature_object)

    return feature_objects


def examples_to_features_lm(examples, label_list, max_seq_len, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_len: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    features = []
    for idx, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
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

        assert len(input_ids) == max_seq_len
        assert len(padding_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(lm_label_ids) == max_seq_len

        if idx < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("padding_mask: %s" % " ".join([str(x) for x in padding_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("LM label: %s " % (lm_label_ids))
            logger.info("Is next sentence label: %s " % (example.label))

        # TODO: adjust to current format of InputFeatures
        features.append(
            InputFeatures(
                input_ids=input_ids,
                padding_mask=padding_mask,
                segment_ids=segment_ids,
                lm_label_ids=lm_label_ids,
                is_next=example.label,
            )
        )
    return features
