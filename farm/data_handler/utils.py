import json
import logging
import os
import random
import tarfile
import tempfile
from itertools import islice

import pandas as pd
from requests import get
from tqdm import tqdm

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
    "conll03entrain": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train",
    "conll03endev": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa",
    "conll03entest": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb",
    "lm_finetune_nips": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz",
    "toxic-comments": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/toxic-comments.tar.gz",
    'cola': "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cola.tar.gz",
}


def read_tsv(filename, rename_columns, quotechar='"', delimiter="\t", skiprows=None, header=0):
    """Reads a tab separated value file. Tries to download the data if filename is not found"""
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename)

    df = pd.read_csv(
        filename,
        sep=delimiter,
        encoding="utf-8",
        quotechar=quotechar,
        dtype=str,
        skiprows=skiprows,
        header=header
    )

    columns = ["text"] + list(rename_columns.keys())
    df = df[columns]
    for source_column, label_name in rename_columns.items():
        df[label_name] = df[source_column].fillna("")
        df.drop(columns=[source_column], inplace=True)
    # convert df to one dict per row
    raw_dict = df.to_dict(orient="records")
    return raw_dict


def read_ner_file(filename, sep="\t", **kwargs):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename)

    f = open(filename, encoding='utf-8')

    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            if len(sentence) > 0:
                data.append({"text": " ".join(sentence), "ner_label": label})
                sentence = []
                label = []
            continue
        splits = line.split(sep)
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        if(label[-1] == ""):
            logger.error(f"The last NER label: '{splits[-1]}'  in your dataset might have been converted incorrectly. Please insert a newline at the end of the file.")
            label[-1] = "O"
        data.append({"text": " ".join(sentence), "ner_label": label})
    return data


def read_squad_file(filename):
    """Read a SQuAD json file"""
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
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
    if "conll03" in taskname:
        # conll03 is copyrighted, but luckily somebody put it on github. Kudos!
        if not os.path.exists(directory):
            os.makedirs(directory)
        for dataset in ["train", "dev", "test"]:
            if "de" in taskname:
                _conll03get(dataset, directory, "de")
            elif "en" in taskname:
                _conll03get(dataset, directory, "en")
            else:
                logger.error("Cannot download {}. Unknown data source.".format(taskname))
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


def _conll03get(dataset, directory, language):
    # open in binary mode
    with open(os.path.join(directory, f"{dataset}.txt"), "wb") as file:
        # get request
        response = get(DOWNSTREAM_TASK_MAP[f"conll03{language}{dataset}"])
        # write to file
        file.write(response.content)


def read_docs_from_txt(filename, delimiter="", encoding="utf-8", max_docs=None):
    """Reads a text file with one sentence per line and a delimiter between docs (default: empty lines) ."""
    if not (os.path.exists(filename)):
        _download_extract_downstream_data(filename)
    all_docs = []
    doc = []
    corpus_lines = 0
    with open(filename, "r", encoding=encoding) as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading Dataset", total=corpus_lines)):
            line = line.strip()
            if line == delimiter:
                if len(doc) > 0:
                    all_docs.append({"doc": doc})
                    doc = []
                    if max_docs:
                        if len(all_docs) >= max_docs:
                            logger.info(f"Reached number of max_docs ({max_docs}). Skipping rest of file ...")
                            break
                else:
                    logger.warning(f"Found empty document in file (line {line_num}). "
                                   f"Make sure that you comply with the format: "
                                   f"One sentence per line and exactly *one* empty line between docs. "
                                   f"You might have multiple subsequent empty lines.")
            else:
                doc.append(line)

        # if last row in file is not empty, we add the last parsed doc manually to all_docs
        if len(doc) > 0:
            if len(all_docs) > 0:
                if all_docs[-1] != doc:
                    all_docs.append({"doc": doc})
            else:
                all_docs.append({"doc": doc})

        if len(all_docs) < 2:
            raise ValueError(f"Found only {len(all_docs)} docs in {filename}). You need at least 2! \n"
                           f"Make sure that you comply with the format: \n"
                           f"-> One sentence per line and exactly *one* empty line between docs. \n"
                           f"You might have a single block of text without empty lines inbetween.")
    return all_docs


def pad(seq, max_seq_len, pad_token, pad_on_left=False):
    ret = seq
    n_required_pad = max_seq_len - len(seq)
    for _ in range(n_required_pad):
        if pad_on_left:
            ret.insert(0, pad_token)
        else:
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


def get_sentence_pair(doc, all_baskets, idx, prob_next_sentence=0.5):
    """
    Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
    from one doc. With 50% the second sentence will be a random one from another doc.

    :param doc: The current document
    :param all_baskets: SampleBaskets containing multiple other docs from which we can sample the second sentence
    if we need a random one.
    :param idx: int, index of sample.
    :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
    """
    sent_1, sent_2 = doc[idx], doc[idx + 1]

    if random.random() > prob_next_sentence:
        label = True
    else:
        sent_2 = _get_random_sentence(all_baskets, forbidden_doc=doc)
        label = False

    assert len(sent_1) > 0
    assert len(sent_2) > 0
    return sent_1, sent_2, label


def _get_random_sentence(all_baskets, forbidden_doc):
    """
    Get random line from another document for nextSentence task.

    :return: str, content of one line
    """
    # Similar to original BERT tf repo: This outer loop should rarely go for more than one iteration for large
    # corpora. However, just to be careful, we try to make sure that
    # the random document is not the same as the document we're processing.
    for _ in range(10):
        rand_doc_idx = random.randrange(len(all_baskets))
        rand_doc = all_baskets[rand_doc_idx]["doc"]

        # check if our picked random doc is really different to our initial doc
        if rand_doc != forbidden_doc:
            rand_sent_idx = random.randrange(len(rand_doc))
            sentence = rand_doc[rand_sent_idx]
            break
    return sentence


def mask_random_words(tokens, vocab, token_groups=None, max_predictions_per_seq=20, masked_lm_prob=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    num_masked. If whole_word_mask is set to true, *all* tokens of a word are either masked or not.
    This option was added by the BERT authors later and showed solid improvements compared to the original objective.
    Whole Word Masking means that if we mask all of the wordpieces corresponding to an original word.
    When a word has been split intoWordPieces, the first token does not have any marker and any subsequence
    tokens are prefixed with ##. So whenever we see the ## token, we
    append it to the previous set of word indexes. Note that Whole Word Masking does *not* change the training code
    at all -- we still predict each WordPiece independently, softmaxed over the entire vocabulary.
    This implementation is mainly a copy from the original code by Google, but includes some simplifications.

    :param tokens: tokenized sentence.
    :type tokens: [str]
    :param vocab: vocabulary for choosing tokens for random masking.
    :type vocab: dict
    :param token_groups: If supplied, only whole groups of tokens get masked. This can be whole words but
    also other types (e.g. spans). Booleans indicate the start of a group.
    :type token_groups: [bool]
    :param max_predictions_per_seq: maximum number of masked tokens
    :type max_predictions_per_seq: int
    :param masked_lm_prob: probability of masking a token
    :type masked_lm_prob: float
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """

    #TODO make special tokens model independent

    # 1. Combine tokens to one group (e.g. all subtokens of a word)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if (token_groups and len(cand_indices) >= 1 and not token_groups[i]):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))

    random.shuffle(cand_indices)
    output_label = [''] * len(tokens)
    num_masked = 0

    # 2. Mask the first groups until we reach the number of tokens we wanted to mask (num_to_mask)
    for index_set in cand_indices:
        if num_masked >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if num_masked + len(index_set) > num_to_mask:
            continue

        for index in index_set:
            prob = random.random()
            num_masked += 1
            original_token = tokens[index]
            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[index] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[index] = random.choice(list(vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label[index] = original_token
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label[index] = "[UNK]"
                logger.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] instead".format(original_token)
                )

    return tokens, output_label


def is_json(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def grouper(iterable, n):
    """
    >>> list(grouper('ABCDEFG'), 3)
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(enumerate(iterable))
    return iter(lambda: list(islice(iterable, n)), [])
