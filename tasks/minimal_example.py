# fmt: off
import logging

import torch

from farm.data_handler.data_bunch import DataBunch
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.modeling.training import (
    Trainer,
    Evaluator,
)
from farm.run_model import initialize_optimizer, calculate_optimization_steps
from farm.utils import set_all_seeds, MLFlowLogger
from farm.data_handler.processor import GNADProcessor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

ml_logger = MLFlowLogger(tracking_uri="http://80.158.39.167:5000/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_minimal_example")

set_all_seeds(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-german-cased",
    do_lower_case=False)

processor = GNADProcessor(tokenizer=tokenizer,
                          max_seq_len=128,
                          data_dir="../data/gnad",
                          train_filename="train.csv")

# Pipeline should also contain metric
data_bunch = DataBunch(
    processor=processor,
    batch_size=32,
    distributed=False)

# Init model
prediction_head = TextClassificationHead(layer_dims=[768, 9])

language_model = Bert.load("bert-base-german-cased")

# TODO where are balance class weights?
model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[prediction_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_sequence"],
    device=device)

# Init optimizer
num_train_optimization_steps = calculate_optimization_steps(
    n_examples=data_bunch.n_samples("train"),
    batch_size=16,
    grad_acc_steps=1,
    n_epochs=1,
    local_rank=-1)

# TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    warmup_proportion=0.1,
    loss_scale=0,
    fp16=False,
    num_train_optimization_steps=num_train_optimization_steps)

# TODO: maybe have a pipeline params object to collapse some of these arguments?
evaluator_dev = Evaluator(
    data_loader=data_bunch.get_data_loader("dev"),
    label_list=processor.label_list,
    device=device,
    metrics=processor.metrics)

evaluator_test = Evaluator(
    data_loader=data_bunch.get_data_loader("test"),
    label_list=processor.label_list,
    device=device,
    metrics=processor.metrics)

trainer = Trainer(
    optimizer=optimizer,
    data_bunch=data_bunch,
    evaluator_dev=evaluator_dev,
    epochs=1,
    n_gpu=1,
    grad_acc_steps=1,
    fp16=False,
    learning_rate=2e-5,  # Why is this also passed to initialize optimizer?
    warmup_linear=warmup_linear,
    evaluate_every=100,
    device=device)

model = trainer.train(model)

model.save("save/model_1")
processor.save("save/model_1")

# FROM HUGGING FACE
# model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
# output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
# output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
# torch.save(model_to_save.state_dict(), output_model_file)
# model_to_save.config.to_json_file(output_config_file)
# tokenizer.save_vocabulary(args.output_dir)


# final evaluation on test set
results = evaluator_test.eval(model)
#TODO this should be executed within the above call
evaluator_test.log_results(results, "Test", trainer.global_step)

# fmt: on
