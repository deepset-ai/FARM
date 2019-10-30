import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.modeling.optimization import initialize_optimizer

from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

set_all_seeds(seed=42)
ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
ml_logger.init_experiment(
    experiment_name="Public_FARM", run_name="Run_minimal_example_lm"
)
##########################
########## Settings
##########################
device, n_gpu = initialize_device_settings(use_cuda=True)
n_epochs = 1
batch_size = 32
evaluate_every = 30
lang_model = "bert-base-cased"

# 1.Create a tokenizer
tokenizer = Tokenizer.from_pretrained(
    pretrained_model_name_or_path=lang_model, do_lower_case=False
)

# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
processor = BertStyleLMProcessor(
    data_dir="../data/lm_finetune_nips", tokenizer=tokenizer, max_seq_len=128, max_docs=30
)
# 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(processor=processor, batch_size=batch_size)

# 4. Create an AdaptiveModel
# a) which consists of a pretrained language model as a basis
language_model = LanguageModel.load(lang_model)
# b) and *two* prediction heads on top that are suited for our task => Language Model finetuning
lm_prediction_head = BertLMHead.load(lang_model)
next_sentence_head = NextSentenceHead.load(lang_model)

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[lm_prediction_head, next_sentence_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_token", "per_sequence"],
    device=device,
)

# 5. Create an optimizer
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    warmup_proportion=0.1,
    n_batches=len(data_silo.loaders["train"]),
    n_epochs=n_epochs,
)

# 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=n_gpu,
    warmup_linear=warmup_linear,
    evaluate_every=evaluate_every,
    device=device,
)

# 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
model = trainer.train(model)

# 8. Hooray! You have a model. Store it:
save_dir = "saved_models/bert-english-lm-tutorial"
model.save(save_dir)
processor.save(save_dir)
