import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import BertLMHead, TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.experiment import initialize_optimizer

from farm.utils import set_all_seeds, MLFlowLogger

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.Create a tokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-cased", do_lower_case=False
)

# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
processor = BertStyleLMProcessor(
    data_dir="../data/finetune_sample", tokenizer=tokenizer, max_seq_len=128
)
# 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(processor=processor, batch_size=32)

# 4. Create an AdaptiveModel
# a) which consists of a pretrained language model as a basis
language_model = Bert.load("bert-base-german-cased")
# b) and *two* prediction heads on top that are suited for our task => Language Model finetuning
lm_prediction_head = BertLMHead(
    embeddings=language_model.model.embeddings,
    hidden_size=language_model.model.config.hidden_size,
)
next_sentence_head = TextClassificationHead(
    layer_dims=[language_model.model.config.hidden_size, 2], loss_ignore_index=-1
)

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
    n_examples=data_silo.n_samples("train"),
    batch_size=16,
    n_epochs=1,
)

# 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=10,
    n_gpu=1,
    warmup_linear=warmup_linear,
    evaluate_every=100,
    device=device,
)

# 7. Let it grow! Watch the tracked metrics live on the public mlflow server: http://80.158.39.167:5000/
model = trainer.train(model)

# 8. Hooray! You have a model. Store it:
save_dir = "save/bert-german-lm-tutorial"
model.save(save_dir)
processor.save(save_dir)
