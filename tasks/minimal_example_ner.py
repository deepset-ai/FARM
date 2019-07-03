import logging
import torch
from opensesame.utils import set_all_seeds
from opensesame.modeling.tokenization import BertTokenizer
from opensesame.data_handler.data_bunch import DataBunch, NewDataBunch
from opensesame.data_handler.input_features import examples_to_features_ner, examples_to_features_sequence
from opensesame.data_handler.utils import read_tsv
from opensesame.data_handler.input_example import create_examples_gnad
from opensesame.data_handler.input_features import examples_to_features_sequence
from opensesame.data_handler.dataset import convert_features_to_dataset

from opensesame.modeling.training import calculate_optimization_steps, initialize_optimizer, Trainer, Evaluator
from opensesame.modeling.language_model import BertModel, Bert
from opensesame.modeling.adaptive_model import AdaptiveModel
from opensesame.modeling.prediction_head import SeqClassificationHead, NERClassificationHead
from opensesame.data_handler.preprocessing_pipeline import PPGNAD, PPCONLL03

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

set_all_seeds(seed=42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-cased-de-2b-end",
                                          do_lower_case=False)


pipeline = PPCONLL03(data_dir="../data/conll03",
                     tokenizer=tokenizer,
                     max_seq_len=128)


# TODO Maybe data_dir should not be an argument here but in pipeline
# Pipeline should also contain metric
data_bunch = NewDataBunch(preprocessing_pipeline=pipeline,
                          batch_size=32,
                          distributed=False)

# Init model
prediction_head = NERClassificationHead(layer_dims=[768, len(pipeline.label_list)])

language_model = Bert.load("bert-base-cased-de-2b-end")
# language_model.save_config("save")

model = AdaptiveModel(language_model=language_model,
                      prediction_head=prediction_head,
                      embeds_dropout_prob=0.1,
                      token_level=pipeline.token_level)
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
                          label_list=pipeline.label_list,
                          device=device,
                          metric=pipeline.metric,
                          output_mode=pipeline.output_mode,
                          token_level=pipeline.token_level)


evaluator_test = Evaluator(data_loader=data_bunch.get_data_loader("test"),
                           label_list=pipeline.label_list,
                           device=device,
                           metric=pipeline.metric,
                           output_mode=pipeline.output_mode,
                           token_level=pipeline.token_level)

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
