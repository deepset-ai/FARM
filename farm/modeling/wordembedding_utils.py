from __future__ import absolute_import, division, print_function, unicode_literals

import io
import os
import json
import logging

import numpy as np

from tqdm import tqdm
from farm.file_utils import load_from_cache
from transformers.tokenization_bert import BertTokenizer
from pathlib import Path


# create dictionaries with links to wordembeddings stored on deepset s3
# the dicts need to be used with HF transformers to use their data + modelling functionality
# language model config
PRETRAINED_CONFIG_ARCHIVE_MAP = {"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/language_model_config.json"}
# tokenization
EMBEDDING_VOCAB_FILES_MAP = {}
EMBEDDING_VOCAB_FILES_MAP["vocab_file"] = {"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vocab.txt"}
MAX_MODEL_INPU_SIZES = {"glove-german-uncased": 10000}
PRETRAINED_INIT_CONFIGURATION = {"glove-german-uncased": {"do_lower_case": False}}
#model
EMBEDDING_MODEL_MAP = {"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vectors.txt"}

#conversion
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]


logger = logging.getLogger(__name__)

def load_embedding_tokenizer(pretrained_model_name_or_path, **kwargs):
    # if the pretrained model points to a file on deepset s3, we need to adjust transformers dictionaries
    if pretrained_model_name_or_path in PRETRAINED_INIT_CONFIGURATION:
        BertTokenizer.pretrained_vocab_files_map["vocab_file"]. \
            update({pretrained_model_name_or_path:EMBEDDING_VOCAB_FILES_MAP["vocab_file"].get(pretrained_model_name_or_path, None)})
        BertTokenizer.max_model_input_sizes. \
            update({pretrained_model_name_or_path:MAX_MODEL_INPU_SIZES.get(pretrained_model_name_or_path,None)})
        BertTokenizer.pretrained_init_configuration. \
            update({pretrained_model_name_or_path:PRETRAINED_INIT_CONFIGURATION.get(pretrained_model_name_or_path,None)})
    ret = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    return ret


def load_model(pretrained_model_name_or_path, **kwargs):
    # loading config
    resolved_config_file = load_from_cache(pretrained_model_name_or_path, PRETRAINED_CONFIG_ARCHIVE_MAP, **kwargs)
    temp = open(resolved_config_file, "r", encoding="utf-8").read()
    config_dict = json.loads(temp)

    # loading vocab
    resolved_vocab_file = load_from_cache(pretrained_model_name_or_path, EMBEDDING_VOCAB_FILES_MAP["vocab_file"], **kwargs)

    # loading model
    resolved_model_file = load_from_cache(pretrained_model_name_or_path, EMBEDDING_MODEL_MAP, **kwargs)

    return config_dict, resolved_vocab_file, resolved_model_file

def load_embedding_vectors(embedding_filename, vocab):
    f = io.open(embedding_filename, 'rt', encoding='utf-8').readlines()

    words_transformed = set()
    repetitions = 0
    embeddings_dimensionality = None
    vectors = {}

    for line in tqdm(f):
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

    # TODO nonzero init of all embeddings, so if it isnt filled it can still learn
    embeddings = np.zeros((len(vocab),embeddings_dimensionality))
    for i, w in enumerate(vocab):
        current = vectors.get(w,np.zeros(embeddings_dimensionality))
        if w not in vectors:
            logger.warning(f"Could not load pretrained embedding for word: {w}")
        embeddings[i,:] = current
    return embeddings

def load_word2vec_vocab(vocab_filename):
    """Loads a vocabulary file into a list."""
    vocab = []
    with open(vocab_filename, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    for l in lines:
        w,c = l.split(" ")
        vocab.append(w.strip())
    return vocab

def convert_WordEmbeddings(embedding_filename, vocab_filename, output_path, language = "English"):
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
    temp_embeddings = load_embedding_vectors(embedding_filename=embedding_filename, vocab=temp_vocab)
    mean_embedding = np.mean(temp_embeddings,axis=0)
    embeddings = np.zeros((temp_embeddings.shape[0]+len(SPECIAL_TOKENS),temp_embeddings.shape[1]))
    for i,tok in enumerate(SPECIAL_TOKENS):
        embeddings[i,:] = mean_embedding
    embeddings[len(SPECIAL_TOKENS):,:] = temp_embeddings

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
    with open(Path(output_path) / "language_model_config.json","w") as file:
        file.write(json.dumps(lm_config,indent=2))

    _save_word2vec_format(fname=str(Path(output_path) / "vectors.txt"),
                          fvocab=str(Path(output_path) / "vocab.txt"),
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
    logger.info(f"storing {vector_size} projection weights into {fname}")
    assert (len(vocab), vector_size) == vectors.shape
    with io.open(fname, 'w') as fout:
        # store in sorted order: most frequent words at the top
        for i,word in enumerate(vocab):
            row = vectors[i,:]
            fout.write(f"{word} {' '.join(repr(val) for val in row)}\n")



if __name__ == "__main__":
    convert_WordEmbeddings(embedding_filename="../../saved_models/glove_normal/vectors.txt",
                           vocab_filename="../../saved_models/glove_normal/vocab.txt",
                           output_path="../../saved_models/glove_converted",
                           language="German")

