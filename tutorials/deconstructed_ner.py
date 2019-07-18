# fmt: off
import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import CONLLProcessor, GermEval14Processor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.experiment import calculate_optimization_steps, initialize_optimizer
from farm.utils import set_all_seeds, MLFlowLogger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ml_logger = MLFlowLogger(tracking_uri="http://80.158.39.167:5000/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_minimal_example_ner")

##########################
########## Settings
##########################
set_all_seeds(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 4
batch_size = 32
evaluate_every = 50
lang_model = "bert-base-german-cased"


tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=lang_model, do_lower_case=False
)

processor = CONLLProcessor(
    tokenizer=tokenizer, max_seq_len=128, data_dir="../data/conll03de"
)

# TODO Maybe data_dir should not be an argument here but in pipeline
# Pipeline should also contain metric
data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

# Init model
prediction_head = TokenClassificationHead(layer_dims=[768, len(processor.label_list)])

language_model = Bert.load(lang_model)
# language_model.save_config("save")

# model = AdaptiveModel.load("save/ner_model_1", device)

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[prediction_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_token"],
    device=device,
)

# Init optimizer
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    warmup_proportion=0.1,
    n_examples=data_silo.n_samples("train"),
    batch_size=batch_size,
    n_epochs=n_epochs,
)


trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=1,
    warmup_linear=warmup_linear,
    evaluate_every=evaluate_every,
    device=device,
)

model = trainer.train(model)

model.save("save/ner_model_1")
processor.save("save/ner_model_1")
