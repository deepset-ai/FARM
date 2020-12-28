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


def test_s3e_fit():
    # small test data
    language_model = Path("samples/s3e/tiny_fasttext_model")
    corpus_path = Path("samples/s3e/tiny_corpus.txt")
    save_dir = Path("testsave/fitted_s3e/")
    do_lower_case = False
    batch_size = 2
    use_gpu = False

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
        lm_output_types=[],
        device=device)

    model, processor, s3e_stats = fit_s3e_on_corpus(processor=processor,
                                                    model=model,
                                                    corpus=corpus_path,
                                                    n_clusters=3,
                                                    pca_n_components=30,
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
                       s3e_stats=s3e_stats, num_processes=0)

    # Input
    basic_texts = [
        {"text": "a man is walking on the street."},
        {"text": "a woman is walking on the street."},
    ]

    # Get embeddings for input text (you can vary the strategy and layer)
    result = inferencer.inference_from_dicts(dicts=basic_texts)
    assert result[0]["context"] == ['a', 'man', 'is', 'walking', 'on', 'the', 'street', '.']
    assert result[0]["vec"][0] - 0.00527727306941057 < 1e-6
    assert result[0]["vec"][-2] - 0.06285100416478565 < 1e-6


def test_load_extract_s3e_embeddings():
    load_dir = Path("samples/s3e/fitted_s3e")
    use_gpu = False
    batch_size = 2

    with open(load_dir / "s3e_stats.pkl", "rb") as f:
        s3e_stats = pickle.load(f)

    # Init inferencer
    inferencer = Inferencer.load(model_name_or_path=load_dir, task_type="embeddings", gpu=use_gpu,
                       batch_size=batch_size, extraction_strategy="s3e", extraction_layer=-1,
                       s3e_stats=s3e_stats, num_processes=0)

    # Input
    basic_texts = [
        {"text": "a man is walking on the street."},
        {"text": "a woman is walking on the street."},
    ]

    # Get embeddings for input text
    result = inferencer.inference_from_dicts(dicts=basic_texts)
    assert result[0]["context"] == ['a', 'man', 'is', 'walking', 'on', 'the', 'street', '.']
    assert result[0]["vec"][0] - 0.00527727306941057 < 1e-6
    assert result[0]["vec"][-2] + 0.06285100416478565 < 1e-6

if __name__ == "__main__":
    test_s3e_fit()
    test_load_extract_s3e_embeddings()