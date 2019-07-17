import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import BertLMHead, TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.experiment import calculate_optimization_steps, initialize_optimizer

from farm.utils import set_all_seeds, MLFlowLogger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

set_all_seeds(seed=42)
ml_logger = MLFlowLogger(tracking_uri="http://80.158.39.167:5000/")
ml_logger.init_experiment(
    experiment_name="Public_FARM", run_name="Run_minimal_example_lm"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-cased", do_lower_case=False
)


processor = BertStyleLMProcessor(
    data_dir="../data/finetune_sample", tokenizer=tokenizer, max_seq_len=128
)

data_silo = DataSilo(processor=processor, batch_size=32, distributed=False)

# Init model
language_model = Bert.load("bert-base-german-cased")

lm_prediction_head = BertLMHead(
    embeddings=language_model.model.embeddings,
    hidden_size=language_model.model.config.hidden_size,
)
next_sentence_head = TextClassificationHead(
    layer_dims=[language_model.model.config.hidden_size, 2], loss_ignore_index=-1
)

model = AdaptiveModel.load("save/lm_model_1", device)

# model = AdaptiveModel(
#     language_model=language_model,
#     prediction_heads=[lm_prediction_head, next_sentence_head],
#     embeds_dropout_prob=0.1,
#     lm_output_types=["per_token", "per_sequence"],
#     device=device,
# )

# Init optimizer
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    warmup_proportion=0.1,
    n_examples=data_silo.n_samples("train"),
    batch_size=16,
    n_epochs=1,
)

trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=10,
    n_gpu=1,
    warmup_linear=warmup_linear,
    evaluate_every=100,
    device=device,
)

model = trainer.train(model)
model.save("save/lm_model_1")
