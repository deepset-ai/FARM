# fmt: off
import logging
from pathlib import Path


from farm.data_handler.processor import InferenceProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds, initialize_device_settings

def embedding_extraction():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    # load from a local path:
    #lang_model = Path("../saved_models/glove-german-uncased")
    # or through s3
    lang_model = "glove-german-uncased" #only glove or word2vec or converted fasttext (fixed vocab) embeddings supported
    do_lower_case = True
    use_amp = None
    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

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


    # Create Inferencer for embedding extraction
    inferencer = Inferencer(
        model=model,
        processor=processor,
        task_type="embeddings"
    )


    # Extract vectors
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]

    result = inferencer.extract_vectors(
        dicts=basic_texts,
        extraction_strategy="cls_token",
        extraction_layer=-1
    )
    print(result)
    inferencer.close_multiprocessing_pool()


if __name__ == "__main__":
    embedding_extraction()

# fmt: on
