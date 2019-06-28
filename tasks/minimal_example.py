import logging
import torch
from opensesame.data_handler.seq_classification import GNADProcessor
from opensesame.utils import set_all_seeds
from opensesame.models.bert.tokenization import BertTokenizer
from opensesame.data_handler.general import NewDataBunch
from opensesame.data_handler.ner import convert_examples_to_features as convert_examples_to_features_ner
from opensesame.data_handler.seq_classification import convert_examples_to_features as convert_examples_to_features_seq

from opensesame.models.bert.training import calculate_optimization_steps, initialize_optimizer, Trainer, Evaluator
from opensesame.models.bert.modeling import FeedForwardFarm, BertModel, BertSeqFarm, BertFarm


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

set_all_seeds(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = "acc"
output_mode = "classification"
token_level = False


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-cased-de-2b-end",
                                          do_lower_case=False)

data_processor = GNADProcessor(data_dir="../data/gnad",
                                  dev_size=0.1,
                                  seed=42)

data_bunch = NewDataBunch.load(data_dir="../data/gnad",
                               data_processor=data_processor,
                               tokenizer=tokenizer,
                               batch_size=16,
                               max_seq_len=256,
                               examples_to_features_fn=convert_examples_to_features_seq)

# Init model
prediction_head = FeedForwardFarm(layer_dims=[768, 9])

language_model = BertFarm.load("bert-base-cased-de-2b-end")
language_model.save_config("save")

model = BertSeqFarm(language_model=language_model,
                    prediction_head=prediction_head,
                    embeds_dropout_prob=0.1)
model.to(device)

# Init optimizer
num_train_optimization_steps = calculate_optimization_steps(n_examples=data_bunch.n_samples("train"),
                                                              batch_size=16,
                                                              grad_acc_steps=1,
                                                              n_epochs=1,
                                                              local_rank=-1)

# TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
optimizer, warmup_linear = initialize_optimizer(model=model,
                                                learning_rate=2e-5,
                                                warmup_proportion=0.1,
                                                loss_scale=0,
                                                fp16=False,
                                                num_train_optimization_steps=num_train_optimization_steps)


evaluator_dev = Evaluator(data_loader=data_bunch.get_data_loader("dev"),
                          label_list=data_bunch.label_list,
                          device=device,
                          metric=metric,
                          output_mode=output_mode,
                          token_level=token_level)


evaluator_test = Evaluator(data_loader=data_bunch.get_data_loader("test"),
                           label_list=data_bunch.label_list,
                           device=device,
                           metric=metric,
                           output_mode=output_mode,
                           token_level=token_level)

trainer = Trainer(optimizer=optimizer,
                  data_bunch=data_bunch,
                  evaluator_dev=evaluator_dev,
                  evaluator_test=evaluator_test,
                  epochs=1,
                  n_gpu=1,
                  grad_acc_steps=1,
                  fp16=False,
                  learning_rate=2e-5,      # Why is this also passed to initialize optimizer?
                  warmup_linear=warmup_linear,
                  evaluate_every=100,
                  device=device)

model = trainer.train(model)

trainer.evaluate_on_test(model)
