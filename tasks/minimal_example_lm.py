import logging

import torch

from farm.data_handler.data_bunch import DataBunch
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import BertLanguageModelHead
from farm.modeling.tokenization import BertTokenizer
from farm.modeling.training import Trainer, Evaluator
from farm.run_model import calculate_optimization_steps, initialize_optimizer

from farm.utils import set_all_seeds

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

set_all_seeds(seed=42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-cased", do_lower_case=False
)


processor = BertStyleLMProcessor(
    data_dir="../data/finetune_sample", tokenizer=tokenizer, max_seq_len=128
)


# TODO Maybe data_dir should not be an argument here but in pipeline
# Pipeline should also contain metric
data_bunch = DataBunch(processor=processor, batch_size=32, distributed=False)

# Init model
language_model = Bert.load("bert-base-cased-de-2b-end")
# language_model.save_config("save")
prediction_head = BertLanguageModelHead(
    embeddings=language_model.model.embeddings,
    hidden_size=language_model.model.config.hidden_size,
)

model = AdaptiveModel(
    language_model=language_model,
    prediction_head=prediction_head,
    embeds_dropout_prob=0.1,
    lm_output_type="both",
)
model.to(device)

# Init optimizer
num_train_optimization_steps = calculate_optimization_steps(
    n_examples=data_bunch.n_samples("train"),
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


evaluator_dev = Evaluator(
    data_loader=data_bunch.get_data_loader("dev"),
    label_list=processor.label_list,
    device=device,
    metric=processor.metric,
    ph_output_type=processor.ph_output_type,
)


evaluator_test = Evaluator(
    data_loader=data_bunch.get_data_loader("test"),
    label_list=processor.label_list,
    device=device,
    metric=processor.metric,
    ph_output_type=processor.ph_output_type,
)

trainer = Trainer(
    optimizer=optimizer,
    data_bunch=data_bunch,
    evaluator_dev=evaluator_dev,
    evaluator_test=evaluator_test,
    epochs=1,
    n_gpu=1,
    grad_acc_steps=1,
    fp16=False,
    learning_rate=2e-5,  # Why is this also passed to initialize optimizer?
    warmup_linear=warmup_linear,
    evaluate_every=100,
    device=device,
)

model = trainer.train(model)

trainer.evaluate_on_test(model)
