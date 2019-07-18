import logging
import os
import json
from requests import get
import tarfile
import tempfile
from tqdm import tqdm
import random
import pandas as pd
import json

from farm.file_utils import http_get

logger = logging.getLogger(__name__)

DOWNSTREAM_TASK_MAP = {
    "gnad": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/gnad.tar.gz",
    "germeval14": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval14.tar.gz",
    "germeval18": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval18.tar.gz",
    "squad20": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    "conll03detrain": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/training_data_for_Stanford_NER/NER-de-train-conll-formated.txt",
    "conll03dedev": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/training_data_for_Stanford_NER/NER-de-dev-conll-formated.txt",
    "conll03detest": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/training_data_for_Stanford_NER/NER-de-test-conll-formated.txt",
}

# TODO skip_first_line is not used here? Do processors expext this to work?
def read_tsv(filename, quotechar='"', delimiter="\t", skiprows=None, columns=None):
    """Reads a tab separated value file. Tries to download the data if filename is not found"""
    if not (os.path.exists(filename)):
        _download_extract_downstream_data(filename)
    df = pd.read_csv(
        filename,
        sep=delimiter,
        encoding="utf-8",
        quotechar=quotechar,
        names=columns,
        skiprows=skiprows,
    )
    if "unused" in df.columns:
        df.drop(columns=["unused"], inplace=True)
    raw_dict = df.to_dict(orient="records")
    return raw_dict


def read_ner_file(filename, **kwargs):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    if not (os.path.exists(filename)):
        _download_extract_downstream_data(filename)
    f = open(filename)

    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            if len(sentence) > 0:
                data.append({"sentence": sentence, "label": label})
                sentence = []
                label = []
            continue
        splits = line.split(" ")
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append({"sentence": sentence, "label": label})
    return data


def read_squad_file(filename):
    """Read a SQuAD json file"""
    if not (os.path.exists(filename)):
        _download_extract_downstream_data(filename)
    with open(filename, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    return input_data


def _download_extract_downstream_data(input_file):
    # download archive to temp dir and extract to correct position
    full_path = os.path.realpath(input_file)
    directory = os.path.dirname(full_path)
    taskname = directory.split("/")[-1]
    datadir = "/".join(directory.split("/")[:-1])
    logger.info(
        "downloading and extracting file {} to dir {}".format(taskname, datadir)
    )
    if "conll03de" in taskname:
        # conll03 is copyrighted, but luckily somebody put it on github. Kudos!
        if not os.path.exists(directory):
            os.mkdir(directory)
        for dataset in ["train", "dev", "test"]:
            _conll03get(dataset, directory)
    elif taskname not in DOWNSTREAM_TASK_MAP:
        logger.error("Cannot download {}. Unknown data source.".format(taskname))
    else:
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here


def _conll03get(dataset, directory):
    # open in binary mode
    with open(os.path.join(directory, f"{dataset}.txt"), "wb") as file:
        # get request
        response = get(DOWNSTREAM_TASK_MAP[f"conll03de{dataset}"])
        # write to file
        file.write(response.content)


def read_docs_from_txt(filename, delimiter="", encoding="utf-8"):
    """Reads a text file with one sentence per line and a delimiter between docs (default: empty lines) ."""
    all_docs = []
    doc = []
    corpus_lines = 0
    # sample_to_doc = []
    with open(filename, "r", encoding=encoding) as f:
        for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
            line = line.strip()
            if line == delimiter:
                all_docs.append({"doc": doc})
                doc = []
                # # remove last added sample because there won't be a subsequent line anymore in the doc
                # sample_to_doc.pop()
            else:
                # store as one sample
                # sample = {"doc_id": len(all_docs), "line": len(doc)}
                # sample_to_doc.append(sample)
                doc.append(line)
                # corpus_lines = corpus_lines + 1

        # if last row in file is not empty
        if all_docs[-1] != doc:
            all_docs.append({"doc": doc})
            # sample_to_doc.pop()

    # data = (all_docs, sample_to_doc)
    return all_docs


def print_example_with_features(
    example, tokens, input_ids, padding_mask, segment_ids, label_ids, initial_mask
):
    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.info("padding_mask: %s" % " ".join([str(x) for x in padding_mask]))
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
    # Inference mode
    if not seq:
        return None
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
    # For inference mode
    if not labels_word:
        return None
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


# def words_to_tokens(words, tokenizer, max_seq_length):
#     tokens_all = []
#     initial_mask = []
#     for w in words:
#         tokens_word = tokenizer.tokenize(w)
#
#         # Sometimes the tokenizer returns no tokens
#         if len(tokens_word) == 0:
#             continue
#
#         n_non_initial_tokens = len(tokens_word) - 1
#         initial_mask += [1]
#         for _ in range(n_non_initial_tokens):
#             initial_mask += [0]
#         tokens_all += tokens_word
#
#     # Clip at max_seq_length. The "-2" is for CLS and SEP token
#     tokens_all = tokens_all[: max_seq_length - 2]
#     initial_mask = initial_mask[: max_seq_length - 2]
#
#     assert len(tokens_all) == len(initial_mask)
#
#     return tokens_all, initial_mask


def get_sentence_pair(doc, all_docs, idx):
    """
    Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
    from one doc. With 50% the second sentence will be a random one from another doc.
    :param idx: int, index of sample.
    :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
    """
    sent_1, sent_2 = doc[idx], doc[idx + 1]

    if random.random() > 0.5:
        label = True
    else:
        sent_2 = _get_random_sentence(all_docs, forbidden_doc=doc)
        label = False

    assert len(sent_1) > 0
    assert len(sent_2) > 0
    return sent_1, sent_2, label


# def _get_subsequent_sentence_pair(all_docs, sample_to_doc, idx):
#     """
#     Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
#     :param idx: int, index of sample.
#     :return: (str, str), two subsequent sentences from corpus
#     """
#     sample = sample_to_doc[idx]
#     t1 = all_docs[sample["doc_id"]][sample["line"]]
#     t2 = all_docs[sample["doc_id"]][sample["line"] + 1]
#
#     # used later to avoid random nextSentence from same doc
#     doc_id = sample["doc_id"]
#     return t1, t2, doc_id


def _get_random_sentence(docs, forbidden_doc):
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

        # check if our picked random doc is really different to our initial doc
        if rand_doc != forbidden_doc:
            sentence = rand_doc[random.randrange(len(rand_doc))]
            break
    return sentence


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


def is_json(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def words_to_tokens(words, word_offsets, tokenizer, max_seq_len):
    tokens = []
    token_offsets = []
    start_of_word = []
    # word_nums = range(0, len(words))
    # initial_mask = []
    for w, w_off in zip(words, word_offsets):
        # Get tokens of single word
        tokens_word = tokenizer.tokenize(w)
        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word
        # get gloabl offset for each token in word + save marker for first tokens of a word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            w_off += len(tok.replace("##", ""))
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)
            # n_non_initial_tokens = len(tokens_word) - 1
        # initial_mask += [1]
        # for _ in range(n_non_initial_tokens):
        #    initial_mask += [0]
    # Clip at max_seq_length. The "-2" is for CLS and SEP token
    # TODO make clipping only dependant on max seq length. E.g. question asnwering has 2 SEP tokens...
    tokens = tokens[: max_seq_len - 2]
    token_offsets = token_offsets[: max_seq_len - 2]
    start_of_word = start_of_word[: max_seq_len - 2]
    # initial_mask = initial_mask[: max_seq_length - 2]
    assert len(tokens) == len(token_offsets) == len(start_of_word)
    return tokens, token_offsets, start_of_word
