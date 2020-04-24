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
                                                    n_clusters=10,
                                                    pca_n_components=30, #300
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
    # lang_model = Path("saved_models/glove-german-uncased")

    lang_model = Path("saved_models/s3e_fasttext")
    corpus_path = Path("data/lm_finetune_nips/train.txt")

    # small test
    lang_model = Path("saved_models/smaller_s3e_fasttext")
    corpus_path = Path("/home/mp/deepset/dev/FARM/test/samples/s3e/tiny_corpus.txt")
    s3e_dir = Path("saved_models/fitted_s3e/")

    fit(language_model=lang_model,
        do_lower_case=False,
        corpus_path=corpus_path,
        save_dir=s3e_dir
        )

    extract_embeddings(load_dir=s3e_dir, use_gpu=False, batch_size=10)