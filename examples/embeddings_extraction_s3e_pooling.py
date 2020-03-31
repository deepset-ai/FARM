import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor, InferenceProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata
from farm.utils import set_all_seeds, initialize_device_settings

from pathlib import Path

logger = logging.getLogger(__name__)
import numpy as np
import io
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter

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
# Construct semantic groups
def semantic_construction(word_weight, cluster_num, word_embs):
    weight_list = list(word_weight.values())
    weight_list = np.array(weight_list)
    print('perform weighted k-means')
    kmeans = KMeans(n_clusters=cluster_num).fit(word_embs, sample_weight=weight_list)

    word_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return word_labels, centroids


def fit_s3e_on_corpus(processor, model, corpus_path, n_clusters=10, eps=1e-3, default_tok_weight=1):
    # Load vocab
    #TODO add threhold for min_count (?)
    word2id = processor.tokenizer.vocab

    # Get word weights
    # token_weights = load_word_weight(path_token_counts, word2id, a=1e-3)
    with open(corpus_path, "r") as f:
        corpus = f.read()
    tokenized_corpus = processor.tokenizer.tokenize(corpus)
    token_counts = dict(Counter(tokenized_corpus))
    n_tokens = sum(token_counts.values())
    token_weights = {}
    for word, id in word2id.items():
        if word in token_counts:
            token_weights[id] = eps / (eps + token_counts[word] / n_tokens)
        else:
            # words that are in vocab but not present in corpus get the default weight
            token_weights[id] = default_tok_weight

    # Load word vectors
    # TODO add normalization
    # TODO normalization that we apply here must also happen at inference time
    # normalized_word_embs = load_wordvec([path_vectors], word2id)
    normalized_word_embs = model.language_model.model.embeddings.cpu().numpy()

    # Construct semantic groups
    token_to_cluster, centroids = semantic_construction(token_weights, n_clusters, normalized_word_embs)

    s3e_stats = {"token_to_cluster": token_to_cluster,
                 "centroids": centroids,
                 "token_weights": token_weights}

    return s3e_stats


def embeddings_extraction():
    set_all_seeds(seed=42)
    batch_size = 32
    use_gpu = False
    lang_model = Path("saved_models/glove-converted-small")     # TODO implement remote language model loading
    do_lower_case = True
    device, n_gpu = initialize_device_settings(use_cuda=use_gpu, use_amp=False)

    # Create a InferenceProcessor
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case)
    processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=128)

    # Create an AdaptiveModel
    language_model = LanguageModel.load(lang_model)
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # Fit S3E on a corpus
    #TODO Load corpus into dataset & extract stats from there
    corpus_path = Path("data/lm_finetune_nips/train.txt")


    s3e_stats = fit_s3e_on_corpus(processor=processor,
                                  model=model,
                                  corpus_path=corpus_path,
                                  n_clusters=10)


    # Load model, tokenizer and processor directly into Inferencer
    # model = Inferencer.load(model_name_or_path=lang_model, task_type="embeddings", gpu=use_gpu, batch_size=batch_size, )
    inferencer = Inferencer(model=model, processor=processor, task_type="embeddings", gpu=use_gpu,
                       batch_size=batch_size, extraction_strategy="s3e", extraction_layer=-1,
                       s3e_stats=s3e_stats)

    # Input
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
        {"text": "Martin Müller spielt Fussball"},
    ]

    # Get embeddings for input text (you can vary the strategy and layer)
    result = inferencer.inference_from_dicts(dicts=basic_texts, )
    print(result)

if __name__ == "__main__":
    embeddings_extraction()