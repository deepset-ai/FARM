import torch

from farm.data_handler.processor import InferenceProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer

from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

##########################
########## Settings
##########################
set_all_seeds(seed=42)
batch_size = 32
use_gpu = True
device, n_gpu = initialize_device_settings(use_cuda=use_gpu)
lang_model = "bert-base-german-cased"

# 1.Create a tokenizer
tokenizer = Tokenizer.from_pretrained(
    pretrained_model_name_or_path=lang_model, do_lower_case=False
)

# 2. Create a lightweight Processor only for inference (no labels, minimal preprocessing)
processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=128)

# 4. Create an AdaptiveModel with  a pretrained language model as a basis
language_model = LanguageModel.load(lang_model)

adaptive_model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[],
    embeds_dropout_prob=0,
    lm_output_types=["per_token", "per_sequence"],
    device=device,
)

# 5. Extract embeddings with model in inference mode
basic_texts = [
    {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
    {"text": "Martin Müller spielt Fussball"},
]

model = Inferencer(adaptive_model, processor, gpu=use_gpu)
result = model.extract_vectors(dicts=basic_texts)
print(result)
