import logging

logger = logging.getLogger(__name__)

import csv
import sys
from tqdm import tqdm
import random


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


def read_docs_from_txt(filename, delimiter="", encoding="utf-8"):
    """Reads a text file with one sentence per line and a delimiter between docs (default: empty lines) ."""
    all_docs = []
    doc = []
    corpus_lines = 0
    sample_to_doc = []
    with open(filename, "r", encoding=encoding) as f:
        for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
            line = line.strip()
            if line == delimiter:
                all_docs.append(doc)
                doc = []
                # remove last added sample because there won't be a subsequent line anymore in the doc
                sample_to_doc.pop()
            else:
                # store as one sample
                sample = {"doc_id": len(all_docs), "line": len(doc)}
                sample_to_doc.append(sample)
                doc.append(line)
                corpus_lines = corpus_lines + 1

        # if last row in file is not empty
        if all_docs[-1] != doc:
            all_docs.append(doc)
            sample_to_doc.pop()

        num_docs = len(all_docs)
    return all_docs, sample_to_doc


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


def get_sentence_pair(docs, sample_to_doc, idx):
    """
    Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
    from one doc. With 50% the second sentence will be a random one from another doc.
    :param idx: int, index of sample.
    :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
    """
    t1, t2, current_doc_id = _get_subsequent_sentence_pair(docs, sample_to_doc, idx)
    if random.random() > 0.5:
        label = 0
    else:
        t2 = _get_random_sentence(docs, current_doc_id)
        label = 1

    assert len(t1) > 0
    assert len(t2) > 0
    return t1, t2, label


def _get_subsequent_sentence_pair(all_docs, sample_to_doc, idx):
    """
    Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
    :param idx: int, index of sample.
    :return: (str, str), two subsequent sentences from corpus
    """
    sample = sample_to_doc[idx]
    t1 = all_docs[sample["doc_id"]][sample["line"]]
    t2 = all_docs[sample["doc_id"]][sample["line"] + 1]

    # used later to avoid random nextSentence from same doc
    doc_id = sample["doc_id"]
    return t1, t2, doc_id


def _get_random_sentence(docs, forbidden_doc_id):
    """
    Get random line from another document for nextSentence task.
    :return: str, content of one line
    """
    # Similar to original BERT tf repo: This outer loop should rarely go for more than one iteration for large
    # corpora. However, just to be careful, we try to make sure that
    # the random document is not the same as the document we're processing.
    for _ in range(10):
        rand_doc_idx = random.randint(0, len(docs) - 1)
        rand_doc = docs[rand_doc_idx]
        line = rand_doc[random.randrange(len(rand_doc))]

        # check if our picked random line is really from another doc like we want it to be
        if rand_doc_idx != forbidden_doc_id:
            break
    return line


def mask_random_words(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                )
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label
