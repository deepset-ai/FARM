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
from collections import OrderedDict
import io
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter


def semantic_construction(word_weight, cluster_num, word_embs):
    weight_list = list(word_weight.values())
    weight_list = np.array(weight_list)
    print('perform weighted k-means')
    kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(word_embs, sample_weight=weight_list)

    word_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return word_labels, centroids


def fit_s3e_on_corpus(processor, model, corpus_path, n_clusters=10, eps=1e-3, default_tok_weight=1):
    #TODO represent unknown tokens with mean vector
    #TODO investigate diff with unknowns (S3e add one vec PER UNK which gets then used in clustering, mean calc etc.)

    # Get tokens of corpus
    with open(corpus_path, "r") as f:
        corpus = f.read()
    #TODO check why EmbeddingTokenizer.tokenize gives many UNK
    tokenized_corpus = tokenize_with_metadata(corpus, processor.tokenizer)["tokens"]
    token_counts = dict(Counter(tokenized_corpus))
    n_tokens = sum(token_counts.values())

    # TODO only temp for testing => save to file
    file = open("saved_models/s3e_fasttext_vocab_counts.txt", "w")
    for k, v in token_counts.items():
        file.write(f"{k} {v}\n")
    file.close()

    # Trim vocab & embeddings to most frequent tokens
    model.language_model.trim_vocab(token_counts, min_threshold=1)
    # TODO find better place for this update
    processor.tokenizer.vocab = model.language_model.model.vocab
    processor.tokenizer.ids_to_tokens = OrderedDict()
    for k, v in processor.tokenizer.vocab.items():
        processor.tokenizer.ids_to_tokens[v] = k

    model.language_model.model.save("saved_models/smaller_s3e_fasttext")

    # Get token weights
    token_weights = {}
    for word, id in processor.tokenizer.vocab.items():
        if word in token_counts:
            token_weights[id] = eps / (eps + token_counts[word] / n_tokens)
        else:
            # words that are in vocab but not present in corpus get the default weight
            token_weights[id] = default_tok_weight

    # Normalize embeddings
    #TODO change n_components back to 300, enable PCA
    model.language_model.normalize_embeddings(zero_mean=True, pca_removal=True, pca_n_components=50, pca_n_top_components=10)

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
    #lang_model = "glove-german-uncased"
    # lang_model = Path("saved_models/s3e_fasttext")
    lang_model = Path("saved_models/smaller_s3e_fasttext")
    # lang_model = Path("saved_models/glove-german-uncased")
    do_lower_case = False
    device, n_gpu = initialize_device_settings(use_cuda=use_gpu, use_amp=False)
    # corpus_path = Path("data/lm_finetune_nips/train.txt")
    corpus_path = Path("/home/mp/deepset/dev/Sentence-Embedding-S3E/custrev_tiny.pos")

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
    s3e_stats = fit_s3e_on_corpus(processor=processor,
                                  model=model,
                                  corpus_path=corpus_path,
                                  n_clusters=3)


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
    result = inferencer.inference_from_dicts(dicts=basic_texts,max_processes=1)
    print(result)

if __name__ == "__main__":
    embeddings_extraction()

    # with open("saved_models/s3e_fasttext/vocab.txt", "w") as wf:
    #     with open("saved_models/s3e_fasttext/vectors.txt", "r") as f:
    #         for l in f:
    #             r = l.split()[0]
    #             wf.write(r)
    #             wf.write("\n")