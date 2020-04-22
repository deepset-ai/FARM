from __future__ import absolute_import, division, print_function, unicode_literals

import io
import json
import logging
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers.tokenization_bert import BertTokenizer

from farm.file_utils import load_from_cache

# create dictionaries with links to wordembeddings stored on deepset s3
# the dicts need to be used with HF transformers to use their data + modelling functionality
# language model config
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "glove-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/language_model_config.json"}
# tokenization
EMBEDDING_VOCAB_FILES_MAP = {}
EMBEDDING_VOCAB_FILES_MAP["vocab_file"] = {
    "glove-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vocab.txt"}
MAX_MODEL_INPU_SIZES = {"glove-german-uncased": 10000}
PRETRAINED_INIT_CONFIGURATION = {"glove-german-uncased": {"do_lower_case": False}}
# model
EMBEDDING_MODEL_MAP = {
    "glove-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vectors.txt",
    "fasttext-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/fasttext-german-uncased/language_model.bin",
}
# conversion
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]

logger = logging.getLogger(__name__)


class Fasttext_converter():
    """
    Class to use fasttext inside FARM by converting embeddings to format usable by preprocessing pipeline.
    Farm needs fixed vocab and embeddings. We can construct a vocab for the data we wish to embed.
    """
    try:
        import fasttext  # fasttext import is optional in requirements. So we just load it when needed.
    except ModuleNotFoundError:
        logger.error("Could not find fasttext. Please install through 'pip install fasttext==0.9.1'.")

    def __init__(self,
                 pretrained_model_name_or_path,
                 do_lower_case,
                 data_path,
                 train_filename,
                 output_path,
                 language=None,
                 sep="\t",
                 text_column_name="text",
                 max_inputdata_rows=None,
                 min_vocab_count=None,
                 max_features=None,
                 ):

        """
        :param pretrained_model_name_or_path: path to local model or pointer to s3
        :param do_lower_case: casing information, should match the vocab
        :param data_path: path to where data is stored
        :param train_filename:
        :param output_path: path where the embeddings (now in word2vec format) are stored
        :param language:
        :param sep: seperator used in train file
        :param text_column_name: column where the text for
        :param max_inputdata_rows: limits the amount of rows to read from data for constructing the vocab
        :param min_vocab_count: when constructing the vocab, words with less than min_vocab_count occurrences are ignored
        :param max_features: maximum number of words to use in vocab
        """

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.do_lower_case = do_lower_case
        self.data_path = data_path
        self.train_filename = train_filename
        self.output_path = output_path
        self.language = language
        self.sep = sep
        self.text_column_name = text_column_name
        self.max_inputdata_rows = max_inputdata_rows
        self.min_vocab_count = min_vocab_count
        self.max_features = max_features

    def convert_on_data(self, **kwargs):
        """
        Function to prepare data by
             - computing a vocab over the data
             - converting each vocab item to a corresponding vector
             - saving vocab and embeddings in word2vec txt format for further processing
        :param kwargs: placeholder for args passed to model loading, like proxy or caching settings
        :return: vocab_counts, dict: dictionary with words and associated counts
        """
        model = self._load_model(**kwargs)

        all_words = self._load_data()

        temp_vocab, vocab_counts = self._create_vocab(all_words=all_words)

        vocab,embeddings = self._create_embeddings(temp_vocab=temp_vocab, model=model)

        self._save(vocab=vocab, embeddings=embeddings)

        return vocab_counts

    def _load_model(self, **kwargs):
        # Model loading
        farm_lm_config = Path(self.pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # from local dir
            config = json.load(open(farm_lm_config, "r"))
            resolved_model_file = str(Path(self.pretrained_model_name_or_path) / config["embeddings_filename"])
        else:
            # from s3 or cache
            resolved_model_file = load_from_cache(self.pretrained_model_name_or_path, EMBEDDING_MODEL_MAP, **kwargs)
        if os.path.isfile(resolved_model_file):
            model = self.fasttext.load_model(resolved_model_file)
        else:
            logger.error(f"Could not load fasttext model at {self.pretrained_model_name_or_path}.")

        return model

    def _load_data(self):
        # Data loading
        df = pd.read_csv(str(self.data_path / self.train_filename), sep=self.sep, nrows=self.max_inputdata_rows)

        if self.text_column_name not in df.columns:
            logger.error(
                f"Cannot find Text column name in the supplied data. Available columsn are {', '.join(df.columns)}.")
        if self.max_inputdata_rows:
            df = df.sample(n=self.max_inputdata_rows)
        texts = df.loc[:, self.text_column_name].values
        all_words = []
        for t in texts:
            if self.do_lower_case:
                t = t.lower()
            words = t.split(" ")
            tokens = []
            for w in words:
                tokens.extend(run_split_on_punc(w))
            all_words.extend(tokens)
        return all_words

    def _create_vocab(self, all_words):
        # Vocab creation
        w, c = np.unique(all_words, return_counts=True)
        if self.min_vocab_count:
            idx = c >= self.min_vocab_count
            w = w[idx]
            c = c[idx]
        if self.max_features:
            max_features_adjusted = self.max_features - len(SPECIAL_TOKENS)
            if w.shape[0] > max_features_adjusted:
                idx = np.argsort(c)[::-1]  # descending order
                w = w[idx[:max_features_adjusted]]
                c = c[idx[:max_features_adjusted]]
        temp_vocab = list(w)
        vocab_counts = dict(zip(temp_vocab, c))
        return temp_vocab, vocab_counts

    def _create_embeddings(self, temp_vocab, model):
        # Embedding creation
        embeddings = np.zeros((len(temp_vocab) + len(SPECIAL_TOKENS), model.get_dimension()))
        for i, w in enumerate(temp_vocab):
            embeddings[i + len(SPECIAL_TOKENS), :] = model.get_word_vector(w)
        mean_embedding = np.mean(embeddings[len(SPECIAL_TOKENS):, :], axis=0)
        for i in range(len(SPECIAL_TOKENS)):
            embeddings[i, :] = mean_embedding
        vocab = SPECIAL_TOKENS + temp_vocab
        return vocab, embeddings

    def _save(self, vocab, embeddings):
        # create config
        lm_config = {
            "embeddings_filename": "vectors.txt",
            "hidden_size": embeddings.shape[1],
            "language": self.language,
            "name": "WordEmbedding_LM",
            "vocab_filename": "vocab.txt",
            "vocab_size": embeddings.shape[0]
        }
        # saving
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        with open(self.output_path / "language_model_config.json", "w") as file:
            file.write(json.dumps(lm_config, indent=2))
        _save_word2vec_format(fname=str(self.output_path / lm_config["embeddings_filename"]),
                              fvocab=str(self.output_path / lm_config["vocab_filename"]),
                              vocab=vocab,
                              vectors=embeddings)


def load_embedding_tokenizer(pretrained_model_name_or_path, **kwargs):
    # if the pretrained model points to a file on deepset s3, we need to adjust transformers dictionaries
    if pretrained_model_name_or_path in PRETRAINED_INIT_CONFIGURATION:
        BertTokenizer.pretrained_vocab_files_map["vocab_file"]. \
            update({pretrained_model_name_or_path: EMBEDDING_VOCAB_FILES_MAP["vocab_file"].get(
            pretrained_model_name_or_path, None)})
        BertTokenizer.max_model_input_sizes. \
            update({pretrained_model_name_or_path: MAX_MODEL_INPU_SIZES.get(pretrained_model_name_or_path, None)})
        BertTokenizer.pretrained_init_configuration. \
            update(
            {pretrained_model_name_or_path: PRETRAINED_INIT_CONFIGURATION.get(pretrained_model_name_or_path, None)})
    ret = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    return ret


def load_model(pretrained_model_name_or_path, **kwargs):
    # loading config
    resolved_config_file = load_from_cache(pretrained_model_name_or_path, PRETRAINED_CONFIG_ARCHIVE_MAP, **kwargs)
    temp = open(resolved_config_file, "r", encoding="utf-8").read()
    config_dict = json.loads(temp)

    # loading vocab
    resolved_vocab_file = load_from_cache(pretrained_model_name_or_path, EMBEDDING_VOCAB_FILES_MAP["vocab_file"],
                                          **kwargs)

    # loading model
    resolved_model_file = load_from_cache(pretrained_model_name_or_path, EMBEDDING_MODEL_MAP, **kwargs)

    return config_dict, resolved_vocab_file, resolved_model_file


def load_embedding_vectors(embedding_file, vocab):
    f = io.open(embedding_file, 'rt', encoding='utf-8').readlines()

    words_transformed = set()
    repetitions = 0
    embeddings_dimensionality = None
    vectors = {}

    for line in tqdm(f, desc="Loading embeddings"):
        line = line.strip()
        if line:
            word, vec = line.split(' ', 1)
            if (word not in words_transformed):  # omit repetitions = speed up + debug
                try:
                    np_vec = np.fromstring(vec, sep=' ')
                    if embeddings_dimensionality is None:
                        if len(np_vec) < 4:  # word2vec includes number of vectors and its dimension as header
                            logger.info("Skipping header")
                            continue
                        else:
                            embeddings_dimensionality = len(np_vec)
                    if len(np_vec) == embeddings_dimensionality:
                        vectors[word] = np_vec
                        words_transformed.add(word)
                except:
                    if logger is not None:
                        logger.debug("Embeddings reader: Could not convert line: {}".format(line))
            else:
                repetitions += 1

    embeddings = np.zeros((len(vocab), embeddings_dimensionality))
    for i, w in enumerate(vocab):
        current = vectors.get(w, np.zeros(embeddings_dimensionality))
        if w not in vectors:
            logger.warning(f"Could not load pretrained embedding for word: {w}")
        embeddings[i, :] = current
    return embeddings


def load_word2vec_vocab(vocab_filename):
    """Loads a vocabulary file into a list."""
    vocab = []
    with open(vocab_filename, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    for l in lines:
        w, c = l.split(" ")
        vocab.append(w.strip())
    return vocab


def convert_WordEmbeddings(embedding_filename, vocab_filename, output_path, language="English"):
    """
    Converts a Wordembedding model in word2vec format to a format that can be used inside FARM
    For compatibility special tokens are added to create embeddings for [CLS], [SEP], [UNK], [PAD] and [MASK]
    The embeddings for these special tokens are the average word embeddings of all other words
    #TODO save model in a more efficient format

    :param vector_filename: word2vec format embeddings. A txt file consisting of space separated word and n (dimension of
    embedding) embedding values for that word
    :type vector_filename: str
    :param vocab_filename: a txt file with each line containing a word and its associated count
    :type vocab_filename: str
    :param output_path: path to store the converted model
    :type output_path: str
    :return:
    """
    # creating vocab
    temp_vocab = load_word2vec_vocab(vocab_filename=vocab_filename)
    vocab = SPECIAL_TOKENS + temp_vocab

    # create embeddings
    temp_embeddings = load_embedding_vectors(embedding_file=embedding_filename, vocab=temp_vocab)
    mean_embedding = np.mean(temp_embeddings, axis=0)
    embeddings = np.zeros((temp_embeddings.shape[0] + len(SPECIAL_TOKENS), temp_embeddings.shape[1]))
    for i, tok in enumerate(SPECIAL_TOKENS):
        embeddings[i, :] = mean_embedding
    embeddings[len(SPECIAL_TOKENS):, :] = temp_embeddings

    # create config
    lm_config = {
        "embeddings_filename": "vectors.txt",
        "hidden_size": embeddings.shape[1],
        "language": language,
        "name": "WordEmbedding_LM",
        "vocab_filename": "vocab.txt",
        "vocab_size": embeddings.shape[0]
    }

    # saving
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(Path(output_path) / "language_model_config.json", "w") as file:
        file.write(json.dumps(lm_config, indent=2))

    _save_word2vec_format(fname=str(Path(output_path) / lm_config["embeddings_filename"]),
                          fvocab=str(Path(output_path) / lm_config["vocab_filename"]),
                          vocab=vocab,
                          vectors=embeddings)


def _save_word2vec_format(fname, vocab, vectors, fvocab):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Code taken and adjusted from gensim: https://github.com/RaRe-Technologies/gensim/blob/ec222e8e3e72608a59805040eadcf5c734a2b96c/gensim/models/utils_any2vec.py#L105

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : list
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    fvocab : str
        File path used to save the vocabulary.
    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    vector_size = vectors.shape[1]
    if fvocab is not None:
        logger.info(f"storing vocabulary in {fvocab}")
        with io.open(fvocab, 'w') as vout:
            for word in vocab:
                vout.write(word + "\n")
    logger.info(f"storing {len(vocab)} projection weights with dimension {vector_size} into {fname}")
    assert (len(vocab), vector_size) == vectors.shape
    with io.open(fname, 'w') as fout:
        # store in sorted order: most frequent words at the top
        for i, word in enumerate(vocab):
            row = vectors[i, :]
            fout.write(f"{word} {' '.join(repr(val) for val in row)}\n")


def run_split_on_punc(text, never_split=None):
    """Splits punctuation on a piece of text.
    Function taken from HuggingFace: transformers.tokenization_bert.BasicTokenizer
    """
    if never_split is not None and text in never_split:
        return [text]
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


if __name__ == "__main__":
    convert_WordEmbeddings(embedding_filename="../../saved_models/glove-normal/vectors.txt",
                           vocab_filename="../../saved_models/glove-normal/vocab.txt",
                           output_path="../../saved_models/glove-converted",
                           language="German")
