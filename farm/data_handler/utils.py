import csv
import logging
import os
import sys
import tarfile
import tempfile

from farm.file_utils import http_get

logger = logging.getLogger(__name__)

DOWNSTREAM_TASK_MAP = {
    "gnad": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/gnad.tar.gz",
    "conll03-de": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/conll03de.tar.gz",
    "germeval14": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval14.tar.gz",
    "germeval18": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval18.tar.gz",
}


def read_tsv(filename, quotechar=None, delimiter="\t"):
    """Reads a tab separated value file. Tries to download the data if filename is not found"""
    if not (os.path.exists(filename)):
        download_extract_downstream_data(filename)
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def read_ner_file(filename, **kwargs):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    if not (os.path.exists(filename)):
        download_extract_downstream_data(filename)
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
    return data


def download_extract_downstream_data(input_file):
    # download archive to temp dir and extract to correct position
    full_path = os.path.realpath(input_file)
    directory = os.path.dirname(full_path)
    taskname = directory.split("/")[-1]
    datadir = "/".join(directory.split("/")[:-1])
    logger.info(
        "downloading and extracting file {} to dir {}".format(taskname, datadir)
    )
    if taskname not in DOWNSTREAM_TASK_MAP:
        logger.error("Cannot download {}. Unknown data source.".format(taskname))
    else:
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here


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
