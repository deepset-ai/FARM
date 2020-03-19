import logging
import io

from farm.infer import Inferencer
from farm.utils import set_all_seeds
from pathlib import Path

logger = logging.getLogger(__name__)
import numpy as np
import io
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load files
def load_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        sentences = [line.split(' ')[:-1] for line in lines]
    sentences = sentences[:-1]

    return sentences


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get word vectors from vocabulary and save as numpy array
def load_wordvec(path_to_vec, word2id):
    N = len(word2id)
    dim = len(path_to_vec) * 300
    word_vec_np = np.zeros((N, dim))

    # For words known
    counts = []
    for i in range(len(path_to_vec)):
        count = 0
        with io.open(path_to_vec[i], 'r', encoding='utf-8') as f:
            # if word2vec or fasttext file : skip first line "next(f)"
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word2id:
                    count = count + 1
                    word_vec_np[word2id[word], i * 300:(i + 1) * 300] = np.fromstring(vec, sep=' ')
        counts.append(count)

        print(path_to_vec[i])
        logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(count, len(word2id)))
        mean_vec = word_vec_np[:, i * 300: (i + 1) * 300].sum(0) / count
        for j in range(N):
            if word_vec_np[j, i * 300] == 0:
                word_vec_np[j, i * 300:(i + 1) * 300] = mean_vec

    print('Unknowns are represented by mean')

    # Pre-processing word embedding: https://arxiv.org/pdf/1808.06305.pdf
    print('pre processing word embedding using https://arxiv.org/pdf/1808.06305.pdf')
    word_vec_np = word_vec_np - np.mean(word_vec_np, 0)
    pca = PCA(n_components=300)
    pca.fit(word_vec_np)

    U1 = pca.components_
    explained_variance = pca.explained_variance_

    # Removing Projections on Top Components
    PVN_dims = 10
    z = []
    for i, x in enumerate(word_vec_np):
        for j, u in enumerate(U1[0:PVN_dims]):
            ratio = (explained_variance[j] - explained_variance[PVN_dims]) / explained_variance[j]
            x = x - ratio * np.dot(u.transpose(), x) * u
        z.append(x)
    word_vec_np = np.asarray(z)

    return word_vec_np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get word weight based on frequency.
def load_word_weight(weightfile, word2id, a=1e-3):
    print('get_word_weights')
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split()
            if (len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value / N)

    # Update for current vocabulary
    weight4ind = {}
    for word, ind in word2id.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0  # for unknown words - how much weight should be given

    return weight4ind


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Construct semantic groups
def semantic_construction(word_weight, cluster_num, word_embs):
    weight_list = list(word_weight.values())
    weight_list = np.array(weight_list)
    print('perform weighted k-means')
    kmeans = KMeans(n_clusters=cluster_num).fit(word_embs, sample_weight=weight_list)

    word_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return word_labels, centroids

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute the sentence embedding
def compute_embedding(args, sentences):
    samples = [sent if sent != [] else ['.'] for sent in sentences]

    sentences_by_id = []

    for sent in samples:
        sentences_by_id.append([args.word2id[word] for word in sent])

    embeddings = []

    # Process each sentence at a time
    for sent_id in sentences_by_id:

        stage_vec = [{}]

        # Original Word Vector (stage_vec[0] = dict with key = word_id, value= word vector)
        for word_id in sent_id:
            stage_vec[-1][word_id] = args.word_vec_np[word_id, :]

        # C (stage_vec[1] = dict with key = cluster num, value= residual vector of all assigned words)
        stage_vec.append({})
        for k, v in stage_vec[-2].items():
            index = args.word_labels[k]

            if index in stage_vec[-1]:
                stage_vec[-1][index].append(stage_vec[-2][k] * args.word_weight[k])
            else:
                stage_vec[-1][index] = []
                stage_vec[-1][index].append(stage_vec[-2][k] * args.word_weight[k])

        # VLAD for each cluster
        for k, v in stage_vec[-1].items():
            # Centroids
            centroid_vec = args.centroids[k]

            # Residual
            v = [wv - centroid_vec for wv in v]
            stage_vec[-1][k] = np.sum(v, 0)

        # Compute Sentence Embedding (weighted mean?)
        sentvec = []
        vec = np.zeros((args.wvec_dim))
        for key, value in stage_vec[0].items():
            vec = vec + value * args.word_weight[key]
        sentvec.append(vec / len(stage_vec[0].keys()))

        # Covariance Descriptor
        matrix = np.zeros((args.cluster_num, args.wvec_dim))
        for j in range(args.cluster_num):
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
        # combining weighted mean of pure wordembeddings + upper triangular of covariance matrix
        sentvec = np.concatenate(sentvec)

        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)

    # Post processing
    if args.postprocessing:
        # Principal Component Removal
        print('post processing sentence embedding using principal component removal')
        svd = TruncatedSVD(n_components=args.postprocessing, n_iter=7, random_state=0)
        svd.fit(embeddings)
        args.svd_comp = svd.components_

        if args.postprocessing == 1:
            embeddings = embeddings - embeddings.dot(args.svd_comp.transpose()) * args.svd_comp
        else:
            embeddings = embeddings - embeddings.dot(args.svd_comp.transpose()).dot(args.svd_comp)

    return embeddings

def fit_s3e_on_corpus(path_corpus, path_vectors, path_word_weights, n_clusters=10):

    # Load text file
    # TODO use FARM processor?
    sentences = load_file(path_corpus)

    # Load dictionary
    # TODO use FARM tokenizer?
    id2word, word2id = create_dictionary(sentences, threshold=5)

    # Load word weights
    # TODO get them actively from corpus?
    word_weight = load_word_weight(path_word_weights, word2id, a=1e-3)

    # Load word vectors
    # TODO use FARM model
    # TODO the normalization that we apply here must also happen at inference time
    normalized_word_embs = load_wordvec([path_vectors], word2id)
    # wvec_dim = word_vec_np.shape[1]

    # Construct semantic groups
    token_to_cluster, centroids = semantic_construction(word_weight, n_clusters, normalized_word_embs)

    s3e_stats = {"token_to_cluster": token_to_cluster,
                 "centroids": centroids,
                 "token_weights": word_weight}

    return s3e_stats


def embeddings_extraction():
    set_all_seeds(seed=42)
    batch_size = 32
    use_gpu = False
    lang_model = "bert-base-german-cased"
    # or local path:
    # lang_model = Path("../saved_models/farm-bert-base-cased-squad2")

    # Load model, tokenizer and processor directly into Inferencer
    model = Inferencer.load(lang_model, task_type="embeddings", gpu=use_gpu, batch_size=batch_size)

    #TODO Load corpus into dataset & extract stats from there

    # Fit S3E on a corpus
    s3e_stats = fit_s3e_on_corpus(path_corpus="../data/lm_finetune_nips/train.txt",
                                  path_vectors="../../Sentence-Embedding-S3E/word_embedding/crawl-300d-2M.vec",
                                  path_word_weights="../../Sentence-Embedding-S3E/word_embedding/enwiki_vocab_min200.txt",
                                  n_clusters=10)

    # Input
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
        {"text": "Martin Müller spielt Fussball"},
    ]

    # Get embeddings for input text (you can vary the strategy and layer)
    result = model.extract_vectors(dicts=basic_texts, extraction_strategy="s3e", extraction_layer=-1, s3e_stats=s3e_stats)
    print(result)

if __name__ == "__main__":
    embeddings_extraction()
