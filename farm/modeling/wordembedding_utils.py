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
from transformers import BertTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from collections import Counter

from farm.file_utils import load_from_cache


# create dictionaries with links to wordembeddings stored on deepset s3
# the dicts need to be used with HF transformers to use their data + modelling functionality
# language model config
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "glove-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/language_model_config.json",
    "glove-english-uncased-6B": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-english-uncased-6B/language_model_config.json",
    "glove-english-cased-840B": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-english-cased-840B/language_model_config.json",
}
# tokenization
EMBEDDING_VOCAB_FILES_MAP = {}
EMBEDDING_VOCAB_FILES_MAP["vocab_file"] = {
    "glove-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vocab.txt",
    "glove-english-uncased-6B": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-english-uncased-6B/vocab.txt",
    "glove-english-cased-840B": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-english-cased-840B/vocab.txt",
}
MAX_MODEL_INPU_SIZES = {
    "glove-german-uncased": 10000,
    "glove-english-uncased-6B": 10000,
    "glove-english-cased-840B": 10000,
}
PRETRAINED_INIT_CONFIGURATION = {"glove-german-uncased": {"do_lower_case": True},
                                 "glove-english-uncased-6B": {"do_lower_case": True},
                                 "glove-english-cased-840B": {"do_lower_case": False}}
# model
EMBEDDING_MODEL_MAP = {
    "glove-german-uncased": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vectors.txt",
    "glove-english-uncased-6B": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-english-uncased-6B/vectors.txt",
    "glove-english-cased-840B": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-english-cased-840B/vectors.txt",
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
        try:
            import fasttext  # fasttext import is optional in requirements. So we just load it when needed.
        except ModuleNotFoundError:
            logger.error("Could not find fasttext. Please install through 'pip install fasttext==0.9.1'.")

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

def s3e_pooling(token_embs, token_ids, token_weights, centroids, token_to_cluster, mask, svd_components=None):
    """
    Pooling of word/token embeddings as described by Wang et al in their paper
    "Efficient Sentence Embedding via Semantic Subspace Analysis"
    (https://arxiv.org/abs/2002.09620)
    Adjusted their implementation from here: https://github.com/BinWang28/Sentence-Embedding-S3E

    This method takes a fitted "s3e model" and token embeddings from a language model and returns sentence embeddings
    using the S3E Method. The model can be fitted via `fit_s3e_on_corpus()`.

    Usage: See `examples/embeddings_extraction_s3e_pooling.py`

    :param token_embs: numpy array of shape (batch_size, max_seq_len, emb_dim) containing the embeddings for each token
    :param token_ids: numpy array of shape (batch_size, max_seq_len) containing the ids for each token in the vocab
    :param token_weights: dict with key=token_id, value= weight in corpus
    :param centroids: numpy array of shape (n_cluster, emb_dim) that describes the centroids of our clusters in the embedding space
    :param token_to_cluster: numpy array of shape (vocab_size, 1) where token_to_cluster[i] = cluster_id that token with id i belongs to
    :param svd_components: Components from a truncated singular value decomposition (SVD, aka LSA) to be
                           removed from the final sentence embeddings in a postprocessing step.
                           SVD must be fit on representative sample of sentence embeddings first and can
                           then be removed from all subsequent embeddings in this function.
                           We expect the sklearn.decomposition.TruncatedSVD.fit(<your_embeddings>)._components to be passed here.
    :return: embeddings matrix of shape (batch_size, emb_dim + (n_clusters*n_clusters+1)/2)
    """

    embeddings = []
    n_clusters = centroids.shape[0]
    emb_dim = token_embs.shape[2]
    # n_tokens = token_embs.shape[1]
    n_samples = token_embs.shape[0]
    # Mask tokens that should be ignored (e.g. Padding, CLS ...)
    token_ids[mask] = -1

    # Process each sentence in the batch at a time
    for sample_idx in range(n_samples):
        stage_vec = [{}]
        # 1) create a dict with key=tok_id, value = embedding
        for tok_idx, tok_id in enumerate(token_ids[sample_idx, :]):
            if tok_id != -1:
                stage_vec[-1][tok_id] = token_embs[sample_idx, tok_idx]

        # 2) create a second dict with key=cluster_id, val=[embeddings] (= C)
        stage_vec.append({})
        for k, v in stage_vec[-2].items():
            cluster = token_to_cluster[k]

            if cluster in stage_vec[-1]:
                stage_vec[-1][cluster].append(stage_vec[-2][k] * token_weights[k])
            else:
                stage_vec[-1][cluster] = []
                stage_vec[-1][cluster].append(stage_vec[-2][k] * token_weights[k])

        # VLAD for each cluster
        for k, v in stage_vec[-1].items():
            # Centroids
            centroid_vec = centroids[k]

            # Residual
            v = [wv - centroid_vec for wv in v]
            stage_vec[-1][k] = np.sum(v, 0)

        # Compute Sentence Embedding (weighted avg, dim = original embedding dim)
        sentvec = []
        vec = np.zeros((emb_dim))
        for key, value in stage_vec[0].items():
            # print(token_weights[key])
            vec = vec + value * token_weights[key]
        sentvec.append(vec / len(stage_vec[0].keys()))

        # Covariance Descriptor (dim = k*(k+1)/2, with k=n_clusters)
        matrix = np.zeros((n_clusters, emb_dim))
        for j in range(n_clusters):
            if j in stage_vec[-1]:
                matrix[j, :] = stage_vec[-1][j]
        matrix_no_mean = matrix - matrix.mean(1)[:, np.newaxis]
        cov = matrix_no_mean.dot(matrix_no_mean.T)

        # Generate Embedding
        iu1 = np.triu_indices(cov.shape[0])
        iu2 = np.triu_indices(cov.shape[0], 1)
        cov[iu2] = cov[iu2] * np.sqrt(2)
        vec = cov[iu1]

        vec = vec / np.linalg.norm(vec)

        sentvec.append(vec)

        # Concatenate weighted avg + covariance descriptors
        sentvec = np.concatenate(sentvec)

        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)

    # Post processing (removal of first principal component)
    if svd_components is not None:
        embeddings = embeddings - embeddings.dot(svd_components.transpose()) * svd_components
    return embeddings


def fit_s3e_on_corpus(processor, model, corpus, n_clusters=10,
                      mean_removal=True, pca_removal=True,
                      pca_n_components=300, pca_n_top_components=10,
                      default_token_weight=1, min_token_occurrences=0,
                      svd_postprocessing=False,
                      use_gpu=False, batch_size=50):
    """
    Pooling of word/token embeddings as described by Wang et al in the paper
    "Efficient Sentence Embedding via Semantic Subspace Analysis"
    (https://arxiv.org/abs/2002.09620)
    Adjusted their implementation from here: https://github.com/BinWang28/Sentence-Embedding-S3E

    This method fits the "model" on a custom corpus. This includes the derivation of token_weights depending on
    token occurences in the corpus, creation of the semantic clusters via k-means and a couple of
    pre-/post-processing steps to normalize the embeddings.

    The resulting objects can be saved or directly passed to the Inferencer to get the actual embeddings for your sentences.
    Note: Some operations like `mean_removal` imply changes on the AdaptiveModel or Processor. That's why we return them.

    :param processor: FARM Processor with a Tokenizer used for reading the corpus (e.g. Inference Processor)
    :param model: FARM AdaptiveModel with an embedding layer in the LM (currently only supporting 'WordEmbedding_LM' as a language model)
    :param corpus: Path to a text file or a str 
    :param n_clusters: Number of clusters for S3E. The more clusters, the higher the dimensionality of the resulting embeddings.
    :param mean_removal: Bool, whether to remove the mean from the token embeddings (preprocessing) 
    :param pca_removal: Bool, whether to remove pca components from the token embeddings (preprocessing)
    :param pca_n_components: int, how many PCA components to fit if `pca_removal` is enabled 
    :param pca_n_top_components: int, how many top PCA components to remove if `pca_removal` is enabled 
    :param default_token_weight: float, what weight to assign for tokens that are in vocab but not in corpus
    :param min_token_occurrences: int, mininum number of token occurrences in the corpus for keeping it in the vocab.
                                  Helps to shrink the model & speed it up.
    :param svd_postprocessing: Bool, whether to remove the top truncated SVD / LSA components from the sentence embeddings (postprocessing).
                               Note: Requires creating all sentence embeddings once for the corpus slowing down this method substantially.
                                     Doesn't impact later inference speed though.
    :param use_gpu: bool, whether to use a GPU
    :param batch_size: int, size of batch for the inferencer (only needed when `svd_postprocessing` is enabled)
    :return: model, processor, s3e_stats
    """

    from farm.infer import Inferencer
    from farm.modeling.tokenization import tokenize_with_metadata

    # Get tokens of corpus
    if isinstance(corpus, Path):
        logger.info("Reading corpus for fitting S3E ")
        with open(corpus, "r") as f:
            corpus = f.read()
    else:
        assert type(corpus) == str, "`corpus` must be of type str or Path()"

    tokenized_corpus = tokenize_with_metadata(corpus, processor.tokenizer)["tokens"]
    token_counts = dict(Counter(tokenized_corpus))
    n_tokens = sum(token_counts.values())

    # Trim vocab & embeddings to most frequent tokens (only to improve speed & ram consumption)
    model.language_model.trim_vocab(token_counts, processor, min_threshold=min_token_occurrences)

    # Normalize embeddings
    model.language_model.normalize_embeddings(zero_mean=mean_removal, pca_removal=pca_removal,
                                              pca_n_components=pca_n_components,
                                              pca_n_top_components=pca_n_top_components)
    normalized_word_embs = model.language_model.model.embeddings.cpu().numpy()

    # Get token weights
    token_weights = {}
    eps = 1e-3
    for word, id in processor.tokenizer.vocab.items():
        if word in token_counts:
            token_weights[id] = eps / (eps + token_counts[word] / n_tokens)
        else:
            # words that are in vocab but not present in corpus get the default weight
            token_weights[id] = default_token_weight

    # Construct Cluster
    weight_list = np.array(list(token_weights.values()))
    logger.info('Creating clusters for S3E embeddings')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(normalized_word_embs, sample_weight=weight_list)

    s3e_stats = {"token_to_cluster": kmeans.labels_,
                 "centroids": kmeans.cluster_centers_,
                 "token_weights": token_weights,
                 "svd_components": None}

    if svd_postprocessing:
        logger.info('Post processing sentence embeddings using principal component removal')

        # Input
        sentences = [{"text": s} for s in corpus.split("\n") if len(s.strip()) > 0]

        # Get embeddings
        try:
            inferencer = Inferencer(model=model, processor=processor, task_type="embeddings", gpu=use_gpu,
                                    batch_size=batch_size, extraction_strategy="s3e", extraction_layer=-1,
                                    s3e_stats=s3e_stats)
            result = inferencer.inference_from_dicts(dicts=sentences)
        finally:
            inferencer.close_multiprocessing_pool()
        sentence_embeddings = [s["vec"] for s in result]
        sentence_embeddings = np.vstack(sentence_embeddings)

        # Principal Component Removal
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(sentence_embeddings)
        s3e_stats["svd_components"] = svd.components_

    return model, processor, s3e_stats


if __name__ == "__main__":
    convert_WordEmbeddings(embedding_filename="../../saved_models/glove-normal/vectors.txt",
                           vocab_filename="../../saved_models/glove-normal/vocab.txt",
                           output_path="../../saved_models/glove-converted",
                           language="German")
