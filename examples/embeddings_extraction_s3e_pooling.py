import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor, InferenceProcessor, Processor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata
from farm.utils import set_all_seeds, initialize_device_settings

from pathlib import Path

logger = logging.getLogger(__name__)
import numpy as np
from collections import OrderedDict
import io
import logging
import pickle

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from collections import Counter


def fit_s3e_on_corpus(processor, model, corpus_path, n_clusters=10, eps=1e-3,
                      center_embeddings=True, pca_removal=True,
                      pca_n_components=300,  pca_n_top_components=10,
                      default_tok_weight=1, svd_postprocessing=False,
                      use_gpu=False, batch_size=10):
    #TODO represent unknown tokens with mean vector
    #TODO investigate diff with unknowns (S3e add one vec PER UNK which gets then used in clustering, mean calc etc.)

    # Get tokens of corpus
    with open(corpus_path, "r") as f:
        corpus = f.read()
    #TODO check why EmbeddingTokenizer.tokenize gives many UNK
    tokenized_corpus = tokenize_with_metadata(corpus, processor.tokenizer)["tokens"]
    token_counts = dict(Counter(tokenized_corpus))
    n_tokens = sum(token_counts.values())

    # Trim vocab & embeddings to most frequent tokens (only to improve speed & ram consumption)
    model.language_model.trim_vocab(token_counts, processor, min_threshold=1)

    # Normalize embeddings
    model.language_model.normalize_embeddings(zero_mean=center_embeddings, pca_removal=pca_removal,
                                              pca_n_components=pca_n_components,
                                              pca_n_top_components=pca_n_top_components)
    normalized_word_embs = model.language_model.model.embeddings.cpu().numpy()

    # Get token weights
    token_weights = {}
    for word, id in processor.tokenizer.vocab.items():
        if word in token_counts:
            token_weights[id] = eps / (eps + token_counts[word] / n_tokens)
        else:
            # words that are in vocab but not present in corpus get the default weight
            token_weights[id] = default_tok_weight

    # Construct Cluster
    weight_list = np.array(list(token_weights.values()))
    logger.info('perform weighted k-means')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(normalized_word_embs, sample_weight=weight_list)

    s3e_stats = {"token_to_cluster": kmeans.labels_,
                 "centroids": kmeans.cluster_centers_,
                 "token_weights": token_weights}

    if svd_postprocessing:
        inferencer = Inferencer(model=model, processor=processor, task_type="embeddings", gpu=use_gpu,
                                batch_size=batch_size, extraction_strategy="s3e", extraction_layer=-1,
                                s3e_stats=s3e_stats)

        # Input
        sentences = [{"text": s} for s in corpus.split("\n")]

        # Get embeddings for input text (you can vary the strategy and layer)
        result = inferencer.inference_from_dicts(dicts=sentences, max_processes=1)
        #TODO n_sentence x emb_dim ndarray
        sentence_embeddings = [s["vec"]]

        # Principal Component Removal
        print('post processing sentence embedding using principal component removal')
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(sentence_embeddings)
        s3e_stats["svd_components"] = svd.components_

    return model, processor, s3e_stats


def fit(language_model, corpus_path, save_dir, do_lower_case, batch_size=4, use_gpu=False):
    # Fit S3E on a corpus
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=use_gpu, use_amp=False)

    # Create a InferenceProcessor
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=language_model, do_lower_case=do_lower_case)
    processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=128)

    # Create an AdaptiveModel
    language_model = LanguageModel.load(language_model)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    model, processor, s3e_stats = fit_s3e_on_corpus(processor=processor,
                                                    model=model,
                                                    corpus_path=corpus_path,
                                                    n_clusters=3,
                                                    pca_n_components=50,
                                                    svd_postprocessing=False)

    # save everything to allow inference without fitting everything again
    model.save(save_dir)
    processor.save(save_dir)
    with open(save_dir / "s3e_stats.pkl", "wb") as f:
        pickle.dump(s3e_stats, f)

    # Load model, tokenizer and processor directly into Inferencer
    inferencer = Inferencer(model=model, processor=processor, task_type="embeddings", gpu=use_gpu,
                       batch_size=batch_size, extraction_strategy="s3e", extraction_layer=-1,
                       s3e_stats=s3e_stats)

    # Input
    basic_texts = [
        {"text": "a man is walking on the street."},
        {"text": "a woman is walking on the street."},
    ]

    # Get embeddings for input text (you can vary the strategy and layer)
    result = inferencer.inference_from_dicts(dicts=basic_texts, max_processes=1)
    print(result)


def extract_embeddings(load_dir, use_gpu, batch_size):
    with open(load_dir / "s3e_stats.pkl", "rb") as f:
        s3e_stats = pickle.load(f)

    # Init inferencer
    inferencer = Inferencer.load(model_name_or_path=load_dir, task_type="embeddings", gpu=use_gpu,
                       batch_size=batch_size, extraction_strategy="s3e", extraction_layer=-1,
                       s3e_stats=s3e_stats)

    # Input
    basic_texts = [
        {"text": "a man is walking on the street."},
        {"text": "a woman is walking on the street."},
    ]

    # Get embeddings for input text
    result = inferencer.inference_from_dicts(dicts=basic_texts, max_processes=1)
    print(result)


if __name__ == "__main__":
    # lang_model = "glove-german-uncased"
    # lang_model = Path("saved_models/s3e_fasttext")
    # lang_model = Path("saved_models/glove-german-uncased")
    lang_model = Path("saved_models/smaller_s3e_fasttext")
    # corpus_path = Path("data/lm_finetune_nips/train.txt")
    s3e_dir = Path("saved_models/fitted_s3e/")

    fit(language_model=lang_model,
        do_lower_case=False,
        corpus_path=Path("/home/mp/deepset/dev/Sentence-Embedding-S3E/custrev_tiny.pos"),
        save_dir=s3e_dir
        )

    extract_embeddings(load_dir=s3e_dir, use_gpu=False, batch_size=10)