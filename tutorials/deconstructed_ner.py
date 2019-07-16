import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import CONLLProcessor
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

set_all_seeds(seed=42)

ml_logger = MLFlowLogger(tracking_uri="http://80.158.39.167:5000/")
ml_logger.init_experiment(
    experiment_name="Public_FARM", run_name="Run_minimal_example_ner"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-german-cased", do_lower_case=False
)

processor = CONLLProcessor(
    tokenizer=tokenizer, max_seq_len=128, data_dir="../data/conll03"
)

# TODO Maybe data_dir should not be an argument here but in pipeline
# Pipeline should also contain metric
data_silo = DataSilo(processor=processor, batch_size=32, distributed=False)

# Init model
prediction_head = TokenClassificationHead(layer_dims=[768, len(processor.label_list)])

language_model = Bert.load("bert-base-german-cased")
# language_model.save_config("save")

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[prediction_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_token"],
    device=device,
)

# Init optimizer
num_train_optimization_steps = calculate_optimization_steps(
    n_examples=data_silo._n_samples("train"),
    batch_size=16,
    grad_acc_steps=1,
    n_epochs=1,
    local_rank=-1,
)

# TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    warmup_proportion=0.1,
    loss_scale=0,
    fp16=False,
    num_train_optimization_steps=num_train_optimization_steps,
)

trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=1,
    n_gpu=1,
    learning_rate=2e-5,  # Why is this also passed to initialize optimizer?
    warmup_linear=warmup_linear,
    evaluate_every=100,
    device=device,
)

model = trainer.train(model)

model.save("save/ner_model_1")
processor.save("save/ner_model_1")
