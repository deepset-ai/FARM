import logging

logger = logging.getLogger(__name__)

import csv
import sys


def read_tsv(filename, quotechar=None, delimiter="\t"):
    """Reads a tab separated value file."""
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, "utf-8") for cell in line)
            lines.append(line)
        return lines


def read_ner_file(filename, **kwargs):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    f = open(filename)

    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(" ")
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


def print_example_with_features(
    example, tokens, input_ids, input_mask, segment_ids, label_ids, initial_mask
):
    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    logger.info("label: %s" % (example.label))
    logger.info("ids  : %s" % (label_ids))
    logger.info("initial_mask: %s" % (initial_mask))


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def add_cls_sep(seq, cls_token, sep_token):
    ret = [cls_token]
    ret += seq
    ret += [sep_token]
    return ret


def pad(seq, max_seq_len, pad_token):
    ret = seq
    n_required_pad = max_seq_len - len(seq)
    for _ in range(n_required_pad):
        ret.append(pad_token)
    return ret


def expand_labels(labels_word, initial_mask, non_initial_token):
    labels_token = []
    word_index = 0
    for im in initial_mask:
        if im:
            # i.e. if token is word initial
            labels_token.append(labels_word[word_index])
            word_index += 1
        else:
            # i.e. token is not the first in the word
            labels_token.append(non_initial_token)

    assert len(labels_token) == len(initial_mask)
    return labels_token


def words_to_tokens(words, tokenizer, max_seq_length):
    tokens_all = []
    initial_mask = []
    for w in words:
        tokens_word = tokenizer.tokenize(w)

        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue

        n_non_initial_tokens = len(tokens_word) - 1
        initial_mask += [1]
        for _ in range(n_non_initial_tokens):
            initial_mask += [0]
        tokens_all += tokens_word

    # Clip at max_seq_length. The "-2" is for CLS and SEP token
    tokens_all = tokens_all[: max_seq_length - 2]
    initial_mask = initial_mask[: max_seq_length - 2]

    assert len(tokens_all) == len(initial_mask)

    return tokens_all, initial_mask
