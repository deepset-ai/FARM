import logging
import pickle
from pathlib import Path

from farm.data_handler.processor import InferenceProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.modeling.wordembedding_utils import fit_s3e_on_corpus

logger = logging.getLogger(__name__)

"""
    Example for generating sentence embeddings via the S3E pooling approach as described by Wang et al in the paper
    "Efficient Sentence Embedding via Semantic Subspace Analysis"
    (https://arxiv.org/abs/2002.09620)
    
    You can use classical models like fasttext, glove or word2vec and apply S3E on top. 
    This can be a powerful benchmark for plain transformer-based embeddings.   

    First, we fit the required stats on a custom corpus. This includes the derivation of token_weights depending on
    token occurences in the corpus, creation of the semantic clusters via k-means and a couple of
    pre-/post-processing steps to normalize the embeddings.
    
    Second, we feed the resulting objects into our Inferencer to extract the actual sentence embeddings for our sentences. 
"""

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
                                                    corpus=corpus_path,
                                                    n_clusters=10,
                                                    pca_n_components=300,
                                                    svd_postprocessing=True,
                                                    min_token_occurrences=1)

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
    result = inferencer.inference_from_dicts(dicts=basic_texts)
    print(result)
    inferencer.close_multiprocessing_pool()


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
    result = inferencer.inference_from_dicts(dicts=basic_texts)
    print(result)
    inferencer.close_multiprocessing_pool()


if __name__ == "__main__":
    lang_model = "glove-english-uncased-6B"
    do_lower_case = True

    # You can download this from:
    # "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz"
    corpus_path = Path("../data/lm_finetune_nips/train.txt")

    s3e_dir = Path("../saved_models/fitted_s3e/")

    fit(language_model=lang_model,
        do_lower_case=do_lower_case,
        corpus_path=corpus_path,
        save_dir=s3e_dir
        )

    extract_embeddings(load_dir=s3e_dir, use_gpu=False, batch_size=10)