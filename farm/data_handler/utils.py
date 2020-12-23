import hashlib
import json
import logging
import os
import random
import tarfile
import tempfile
import string
from contextlib import ExitStack
from itertools import islice
from pathlib import Path

import pandas as pd
from requests import get
from tqdm import tqdm
from typing import List

from farm.file_utils import http_get
from farm.modeling.tokenization import tokenize_with_metadata

logger = logging.getLogger(__name__)

DOWNSTREAM_TASK_MAP = {
    "gnad": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/gnad.tar.gz",
    "germeval14": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval14.tar.gz",

    # only has train.tsv and test.tsv dataset - no dev.tsv
    "germeval18": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval18.tar.gz",

    "squad20": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    "covidqa": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz",

    "conll03detrain": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train",
    "conll03dedev": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa", #https://www.clips.uantwerpen.be/conll2003/ner/000README says testa is dev data
    "conll03detest": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb",
    "conll03entrain": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train",
    "conll03endev": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa",
    "conll03entest": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb",
    "cord_19": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cord_19.tar.gz",
    "lm_finetune_nips": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz",
    "toxic-comments": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/toxic-comments.tar.gz",
    'cola': "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cola.tar.gz",
    "asnq_binary": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/asnq_binary.tar.gz",
    "germeval17": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval17.tar.gz",
    "natural_questions": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/natural_questions.tar.gz",

}

def read_tsv(filename, rename_columns, quotechar='"', delimiter="\t", skiprows=None, header=0, proxies=None, max_samples=None):
    """Reads a tab separated value file. Tries to download the data if filename is not found"""

    # get remote dataset if needed
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies=proxies)

    # read file into df - but only read those cols we need
    columns_needed = list(rename_columns.keys())
    df = pd.read_csv(
        filename,
        sep=delimiter,
        encoding="utf-8",
        quotechar=quotechar,
        dtype=str,
        skiprows=skiprows,
        header=header,
        usecols=columns_needed,
    )
    if max_samples:
        df = df.sample(max_samples)

    # let's rename our target columns to the default names FARM expects:
    # "text": contains the text
    # "text_classification_label": contains a label for text classification
    df.rename(columns=rename_columns, inplace=True)
    df.fillna("", inplace=True)

    # convert df to one dict per row
    raw_dict = df.to_dict(orient="records")
    return raw_dict

def read_tsv_sentence_pair(filename, rename_columns, delimiter="\t", skiprows=None, header=0, proxies=None, max_samples=None):
    """Reads a tab separated value file. Tries to download the data if filename is not found"""

    # get remote dataset if needed
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies=proxies)

    # TODO quote_char was causing trouble for the asnq dataset so it has been removed - see if there's a better solution
    df = pd.read_csv(
        filename,
        sep=delimiter,
        encoding="utf-8",
        dtype=str,
        skiprows=skiprows,
        header=header
    )
    if max_samples:
        df = df.sample(max_samples)

    # let's rename our target columns to the default names FARM expects:
    # "text": contains the text
    # "text_classification_label": contains a label for text classification
    columns = ["text"] + ["text_b"] + list(rename_columns.keys())
    df = df[columns]
    for source_column, label_name in rename_columns.items():
        df[label_name] = df[source_column].fillna("")
        df.drop(columns=[source_column], inplace=True)
    # convert df to one dict per row
    raw_dict = df.to_dict(orient="records")
    return raw_dict

def read_jsonl(file, proxies=None):
    # get remote dataset if needed
    if not (os.path.exists(file)):
        logger.info(f" Couldn't find {file} locally. Trying to download ...")
        _download_extract_downstream_data(file, proxies=proxies)
    dicts = [json.loads(l) for l in open(file, encoding="utf-8")]
    return dicts

def read_ner_file(filename, sep="\t", proxies=None):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    # checks for correct separator
    if "conll03-de" in str(filename):
        if sep != " ":
            logger.error(f"Separator {sep} for dataset German CONLL03 does not match the requirements. Setting seperator to whitespace")
            sep = " "
    if "germeval14" in str(filename):
        if sep != "\t":
            logger.error(f"Separator {sep} for dataset GermEval14 de does not match the requirements. Setting seperator to tab")
            sep = "\t"

    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies)
    if "conll03-de" in str(filename):
        f = open(filename, encoding='cp1252')
    else:
        f = open(filename, encoding='utf-8')

    data = []
    sentence = []
    label = []
    for line in f:
        if line.startswith("#"):
            continue
        if len(line) == 0 or "-DOCSTART-" in line or line[0] == "\n":
            if len(sentence) > 0:
                if "conll03" in str(filename):
                    _convertIOB1_to_IOB2(label)
                if "germeval14" in str(filename):
                    label = _convert_germeval14_labels(label)
                data.append({"text": " ".join(sentence), "ner_label": label})
                sentence = []
                label = []
            continue
        splits = line.split(sep)

        # adjusting to data format in Germeval14
        # Germeval14 has two levels of annotation. E.g. "Univerität Berlin" is both ORG and LOC. We only take the first level.
        if "germeval14" in str(filename):
            sentence.append(splits[1])
            label.append(splits[-2])
        else:
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

    # handling end of file, adding the last sentence to data
    if len(sentence) > 0:
        if(label[-1] == ""):
            logger.error(f"The last NER label: '{splits[-1]}'  in your dataset might have been converted incorrectly. Please insert a newline at the end of the file.")
            label[-1] = "O"

        if "conll03-de" in str(filename):
            _convertIOB1_to_IOB2(label)
        if "germeval14" in str(filename):
            label = _convert_germeval14_labels(label)
        data.append({"text": " ".join(sentence), "ner_label": label})
    return data

def read_dpr_json(file, max_samples=None, proxies=None):
    """
    Reads a Dense Passage Retrieval (DPR) data file in json format and returns a list of dictionaries.

    :param file: filename of DPR data in json format

    Returns:
        list of dictionaries: List[dict]
        each dictionary: {
                    "query": str -> query_text
                    "passages": List[dictionaries] -> [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
                                {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
                                ...]
                    }
        example:
                ["query": 'who sings does he love me with reba'
                "passages" : [{'title': 'Does He Love You',
                    'text': 'Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba\'s album "Greatest Hits Volume Two". It is one of country music\'s several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members',
                    'label': 'positive',
                    'external_id': '11828866'},
                    {'title': 'When the Nightingale Sings',
                    'text': "When the Nightingale Sings When The Nightingale Sings is a Middle English poem, author unknown, recorded in the British Library's Harley 2253 manuscript, verse 25. It is a love poem, extolling the beauty and lost love of an unknown maiden. When þe nyhtegale singes þe wodes waxen grene.<br> Lef ant gras ant blosme springes in aueryl y wene,<br> Ant love is to myn herte gon wiþ one spere so kene<br> Nyht ant day my blod hit drynkes myn herte deþ me tene. Ich have loved al þis er þat y may love namore,<br> Ich have siked moni syk lemmon for",
                    'label': 'hard_negative',
                    'external_id': '10891637'}]
                ]

    """
    # get remote dataset if needed
    if not (os.path.exists(file)):
        logger.info(f" Couldn't find {file} locally. Trying to download ...")
        _download_extract_downstream_data(file, proxies=proxies)
    dicts = json.load(open(file))
    if max_samples:
        dicts = random.sample(dicts, min(max_samples, len(dicts)))
    # convert DPR dictionary to standard dictionary
    query_json_keys = ["question", "questions", "query"]
    positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
    hard_negative_json_keys = ["hard_negative_contexts", "hard_negative_ctxs", "hard_negative_context", "hard_negative_ctx"]
    standard_dicts = []
    for dict in dicts:
        sample = {}
        passages = []
        for key, val in dict.items():
            if key in query_json_keys:
                sample["query"] = val
            elif key in positive_context_json_keys+hard_negative_json_keys:
                for passage in val:
                    passages.append({
                        "title": passage["title"],
                        "text": passage["text"],
                        "label": "positive" if key in positive_context_json_keys else "hard_negative",
                        "external_id": passage["passage_id"]
                        })
        sample["passages"] = passages
        standard_dicts.append(sample)
    return standard_dicts

def _convert_germeval14_labels(tags: List[str]):
    newtags = []
    for tag in tags:
        tag = tag.replace("part","")
        tag = tag.replace("deriv","")
        newtags.append(tag)
    return newtags



def _convertIOB1_to_IOB2(tags: List[str]):
    """
    script taken from: https://gist.github.com/allanj/b9bd448dc9b70d71eb7c2b6dd33fe4ef
    IOB1:  O I I B I
    IOB2:  O B I B I
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def read_squad_file(filename, proxies=None):
    """Read a SQuAD json file"""
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies)
    with open(filename, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    return input_data

def write_squad_predictions(predictions, out_filename, predictions_filename=None):
    predictions_json = {}
    for x in predictions:
        for p in x["predictions"]:
            if p["answers"][0]["answer"] is not None:
                predictions_json[p["question_id"]] = p["answers"][0]["answer"]
            else:
                predictions_json[p["question_id"]] = "" #convert No answer = None to format understood by the SQuAD eval script

    if predictions_filename:
        dev_labels = {}
        temp = json.load(open(predictions_filename, "r"))
        for d in temp["data"]:
            for p in d["paragraphs"]:
                for q in p["qas"]:
                    if q.get("is_impossible",False):
                        dev_labels[q["id"]] = "is_impossible"
                    else:
                        dev_labels[q["id"]] = q["answers"][0]["text"]
        not_included = set(list(dev_labels.keys())) - set(list(predictions_json.keys()))
        if len(not_included) > 0:
            logger.info(f"There were missing predicitons for question ids: {list(not_included)}")
        for x in not_included:
            predictions_json[x] = ""

    # os.makedirs("model_output", exist_ok=True)
    # filepath = Path("model_output") / out_filename
    json.dump(predictions_json, open(out_filename, "w"))
    logger.info(f"Written Squad predictions to: {out_filename}")

def _get_md5checksum(fname):
    # solution from stackoverflow: https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _download_extract_downstream_data(input_file, proxies=None):
    # download archive to temp dir and extract to correct position
    full_path = Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info(
        "downloading and extracting file {} to dir {}".format(taskname, datadir)
    )
    if "conll03-" in taskname:
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
        if os.name == "nt":  # make use of NamedTemporaryFile compatible with Windows
            delete_tmp_file = False
        else:
            delete_tmp_file = True
        with tempfile.NamedTemporaryFile(delete=delete_tmp_file) as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible

            # checking files for correctness with md5sum.
            if("germeval14" in taskname):
                if "2c9d5337d7a25b9a4bf6f5672dd091bc" != _get_md5checksum(temp_file.name):
                    logger.error(f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")
            elif "germeval18" in taskname:
                if "23244fa042dcc39e844635285c455205" != _get_md5checksum(temp_file.name):
                    logger.error(f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")
            elif "gnad" in taskname:
                if "ef62fe3f59c1ad54cf0271d8532b8f22" != _get_md5checksum(temp_file.name):
                    logger.error(f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")
            elif "germeval17" in taskname:
                if "f1bf67247dcfe7c3c919b7b20b3f736e" != _get_md5checksum(temp_file.name):
                    logger.error(f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here


def _conll03get(dataset, directory, language):
    # open in binary mode
    with open(directory / f"{dataset}.txt", "wb") as file:
        # get request
        response = get(DOWNSTREAM_TASK_MAP[f"conll03{language}{dataset}"])
        # write to file
        file.write(response.content)

    # checking files for correctness with md5sum.
    if f"conll03{language}{dataset}" == "conll03detrain":
        if "ae4be68b11dc94e0001568a9095eb391" != _get_md5checksum(str(directory / f"{dataset}.txt")):
            logger.error(
                f"Someone has changed the file for conll03detrain. This data was collected from an external github repository.\n"
                f"Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")
    elif f"conll03{language}{dataset}" == "conll03detest":
        if "b8514f44366feae8f317e767cf425f28" != _get_md5checksum(str(directory / f"{dataset}.txt")):
            logger.error(
                f"Someone has changed the file for conll03detest. This data was collected from an external github repository.\n"
                f"Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")
    elif f"conll03{language}{dataset}" == "conll03entrain":
        if "11a942ce9db6cc64270372825e964d26" != _get_md5checksum(str(directory / f"{dataset}.txt")):
            logger.error(
                f"Someone has changed the file for conll03entrain. This data was collected from an external github repository.\n"
                f"Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py")



def read_docs_from_txt(filename, delimiter="", encoding="utf-8", max_docs=None, proxies=None, disable_tqdm=True):
    """Reads a text file with one sentence per line and a delimiter between docs (default: empty lines) ."""
    if not (os.path.exists(filename)):
        _download_extract_downstream_data(filename, proxies)

    doc_count = 0
    doc = []
    prev_doc = None
    corpus_lines = 0

    with open(filename, "r", encoding=encoding) as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading Dataset", total=corpus_lines, disable=disable_tqdm)):
            line = line.strip()
            if line == delimiter:
                if len(doc) > 0:
                    yield {"doc": doc}
                    doc_count += 1
                    prev_doc = doc
                    doc = []
                    if max_docs:
                        if doc_count >= max_docs:
                            logger.info(f"Reached number of max_docs ({max_docs}). Skipping rest of file ...")
                            break
                else:
                    logger.warning(f"Found empty document in '{filename}' (line {line_num}). "
                                   f"Make sure that you comply with the format: "
                                   f"One sentence per line and exactly *one* empty line between docs. "
                                   f"You might have multiple subsequent empty lines.")
            else:
                doc.append(line)

        # if last row in file is not empty, we add the last parsed doc manually to all_docs
        if len(doc) > 0:
            if doc_count > 0:
                if doc != prev_doc:
                    yield {"doc": doc}
                    doc_count += 1
            else:
                yield {"doc": doc}
                doc_count += 1

        if doc_count < 2:
            raise ValueError(f"Found only {doc_count} docs in {filename}). You need at least 2! \n"
                           f"Make sure that you comply with the format: \n"
                           f"-> One sentence per line and exactly *one* empty line between docs. \n"
                           f"You might have a single block of text without empty lines inbetween.")


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
    sentence = None
    for _ in range(100):
        rand_doc_idx = random.randrange(len(all_baskets))
        rand_doc = all_baskets[rand_doc_idx]

        # check if our picked random doc is really different to our initial doc
        if rand_doc != forbidden_doc:
            rand_sent_idx = random.randrange(len(rand_doc))
            sentence = rand_doc[rand_sent_idx]
            break
    if sentence is None:
        raise Exception("Failed to pick out a suitable random substitute for next sentence")
    return sentence



    # return sequence_a, sequence_b, sample_in_clear_text, num_unused_segments


def _get_random_doc(all_baskets, forbidden_doc):
    random_doc = None
    for _ in range(100):
        rand_doc_idx = random.randrange(len(all_baskets))
        random_doc = all_baskets[rand_doc_idx]["doc"]

        # check if random doc is different from initial doc
        if random_doc != forbidden_doc:
            break

    if random_doc is None:
        raise Exception("Failed to pick out a suitable random substitute for next sequence")
    return random_doc


def join_sentences(sequence):
    """
    Takes a list of subsequent, tokenized sentences and puts them together into one sequence.
    :param sequence: List of tokenized sentences.
    :type sequence: [dict]
    :return: Tokenized sequence. (Dict with keys 'tokens', 'offsets' and 'start_of_word')
    """
    sequence_joined = {
        "tokens" : [],
        "offsets" : [],
        "start_of_word" : []
    }
    last_offset = 0
    for sentence in sequence:
        sequence_joined["tokens"].extend(sentence["tokens"])
        sequence_joined["start_of_word"].extend(sentence["start_of_word"])
        # get offsets right
        current_offsets = [offset + last_offset for offset in sentence["offsets"]]
        sequence_joined["offsets"].extend(current_offsets)
        last_offset += sentence["offsets"][-1] + 2

    return sequence_joined





def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False


def grouper(iterable, n, worker_id=0, total_workers=1):
    """
    Split an iterable into a list of n-sized chunks. Each element in the chunk is a tuple of (index_num, element).

    Example:

    >>> list(grouper('ABCDEFG', 3))
    [[(0, 'A'), (1, 'B'), (2, 'C')], [(3, 'D'), (4, 'E'), (5, 'F')], [(6, 'G')]]



    Use with the StreamingDataSilo

    When StreamingDataSilo is used with multiple PyTorch DataLoader workers, the generator
    yielding dicts(that gets converted to datasets) is replicated across the workers.

    To avoid duplicates, we split the dicts across workers by creating a new generator for
    each worker using this method.

    Input --> [dictA, dictB, dictC, dictD, dictE, ...] with total worker=3 and n=2

    Output for worker 1: [(dictA, dictB), (dictG, dictH), ...]
    Output for worker 2: [(dictC, dictD), (dictI, dictJ), ...]
    Output for worker 3: [(dictE, dictF), (dictK, dictL), ...]

    This method also adds an index number to every dict yielded.

    :param iterable: a generator object that yields dicts
    :type iterable: generator
    :param n: the dicts are grouped in n-sized chunks that gets converted to datasets
    :type n: int
    :param worker_id: the worker_id for the PyTorch DataLoader
    :type worker_id: int
    :param total_workers: total number of workers for the PyTorch DataLoader
    :type total_workers: int
    """
    # TODO make me comprehensible :)
    def get_iter_start_pos(gen):
        start_pos = worker_id * n
        for i in gen:
            if start_pos:
                start_pos -= 1
                continue
            yield i

    def filter_elements_per_worker(gen):
        x = n
        y = (total_workers - 1) * n
        for i in gen:
            if x:
                yield i
                x -= 1
            else:
                if y != 1:
                    y -= 1
                    continue
                else:
                    x = n
                    y = (total_workers - 1) * n

    iterable = iter(enumerate(iterable))
    iterable = get_iter_start_pos(iterable)
    if total_workers > 1:
        iterable = filter_elements_per_worker(iterable)

    return iter(lambda: list(islice(iterable, n)), [])


def split_file(filepath, output_dir, docs_per_file=1_000, delimiter="", encoding="utf-8"):
    total_lines = sum(1 for line in open(filepath, encoding=encoding))
    output_file_number = 1
    doc_count = 0
    lines_to_write = []
    with ExitStack() as stack:
        input_file = stack.enter_context(open(filepath, 'r', encoding=encoding))
        for line_num, line in enumerate(tqdm(input_file, desc="Splitting file ...", total=total_lines)):
            lines_to_write.append(line)
            if line.strip() == delimiter:
                doc_count += 1
                if doc_count % docs_per_file == 0:
                    filename = output_dir / f"part_{output_file_number}"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    write_file = stack.enter_context(open(filename, 'w+', encoding=encoding, buffering=10 * 1024 * 1024))
                    write_file.writelines(lines_to_write)
                    write_file.close()
                    output_file_number += 1
                    lines_to_write = []

        if lines_to_write:
            filename = output_dir / f"part_{output_file_number}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            write_file = stack.enter_context(open(filename, 'w+', encoding=encoding, buffering=10 * 1024 * 1024))
            write_file.writelines(lines_to_write)
            write_file.close()

    logger.info(f"The input file {filepath} is split in {output_file_number} parts at {output_dir}.")


def generate_tok_to_ch_map(text):
    """ Generates a mapping from token to character index when a string text is split using .split()
    TODO e.g."""
    map = [0]
    follows_whitespace = False
    for i, ch in enumerate(text):
        if follows_whitespace:
            if ch not in string.whitespace:
                map.append(i)
                follows_whitespace = False
        else:
            if ch in string.whitespace:
                follows_whitespace = True
    return map


def split_with_metadata(text):
    """" Splits a string text by whitespace and also returns indexes which is a mapping from token index
    to character index"""
    split_text = text.split()
    indexes = generate_tok_to_ch_map(text)
    assert len(split_text) == len(indexes)
    return split_text, indexes
