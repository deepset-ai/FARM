from farm.infer import Inferencer
from farm.utils import set_all_seeds
from pathlib import Path

def embeddings_extraction():
    set_all_seeds(seed=42)
    batch_size = 32
    use_gpu = False
    lang_model = "bert-base-german-cased"
    # or local path:
    # lang_model = Path("../saved_models/farm-bert-base-cased-squad2")

    # Input
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
        {"text": "Martin Müller spielt Fussball"},
    ]

    # Load model, tokenizer and processor directly into Inferencer
    model = Inferencer.load(lang_model, task_type="embeddings", gpu=use_gpu, batch_size=batch_size,
                            extraction_strategy="reduce_mean", extraction_layer=-2, num_processes=0)

    # Get embeddings for input text (you can vary the strategy and layer)
    result = model.inference_from_dicts(dicts=basic_texts)
    print(result)
    model.close_multiprocessing_pool()


if __name__ == "__main__":
    embeddings_extraction()
