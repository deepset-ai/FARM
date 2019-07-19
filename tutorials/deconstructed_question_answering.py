# fmt: off
import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import QuestionAnsweringHead
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
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")

##########################
########## Settings
##########################
set_all_seeds(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
batch_size = 24*4
n_epochs = 3
evaluate_every = 200
n_gpu = 4
base_LM_model = "bert-base-cased"
train_filename="train-v2.0.json"
dev_filename="dev-v2.0.json"
save_dir = "../save/qa_model_full"


tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=base_LM_model, do_lower_case=False
)

processor = SquadProcessor(
    tokenizer=tokenizer,
    max_seq_len=256,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=None,
    data_dir="../data/squad20",
)

# TODO Maybe data_dir should not be an argument here but in pipeline
# Pipeline should also contain metric
data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

# Init model
prediction_head = QuestionAnsweringHead(layer_dims=[768, len(processor.label_list)])

language_model = Bert.load(base_LM_model)
# language_model.save_config("save")

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
    learning_rate=1e-5,
    warmup_proportion=0.2,
    n_examples=data_silo.n_samples("train"),
    batch_size=batch_size,
    n_epochs=n_epochs,
)


trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=n_gpu,
    warmup_linear=warmup_linear,
    evaluate_every=evaluate_every,
    device=device,
)

model = trainer.train(model)

model.save(save_dir)
processor.save(save_dir)
