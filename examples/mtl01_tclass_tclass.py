#!/usr/bin/env python
# coding: utf-8


import logging
import torch
import farm
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.infer import Inferencer
from farm.eval import Evaluator



print("Pytorch version:", torch.__version__)
print("CUDA library in pytorch:", torch.version.cuda)
print("FARM version:", farm.__version__)


#logger = MLFlowLogger(tracking_uri="mlflowlog01")
#logger.init_experiment(experiment_name="farm_building_blocks", run_name="tutorial")


logging.basicConfig(level="INFO")
logger = logging.getLogger(name="mtl01-train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Devices available: {}".format(device))


LANG_MODEL = "bert-base-german-cased" 
BATCH_SIZE = 32
MAX_SEQ_LEN = 128
EMBEDS_DROPOUT_PROB = 0.1
LEARNING_RATE = 3e-5
MAX_N_EPOCHS = 6
N_GPU = 1
EVAL_EVERY = 70
DATA_DIR = "../data/germeval18"
PREDICT = "both"                # coarse, fine or both
DO_ROUND_ROBIN = False            # round robin training of heads?


logger.info("Loading Tokenizer")
tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=LANG_MODEL,
    do_lower_case=False)


LABEL_LIST_COARSE = ["OTHER", "OFFENSE"]
LABEL_LIST_FINE = ["OTHER", "ABUSE", "INSULT", "PROFANITY"]

metrics_fine = "f1_macro"
metrics_coarse = "f1_macro"


processor = TextClassificationProcessor(tokenizer=tokenizer,
                                        max_seq_len=MAX_SEQ_LEN,
                                        data_dir=DATA_DIR,
                                        dev_split=0.1,
                                        text_column_name="text",
                                        )
prediction_heads = []
if PREDICT == "coarse" or PREDICT == "both":
    processor.add_task(name="coarse",
                           task_type="classification",
                           label_list=LABEL_LIST_COARSE,
                           metric=metrics_coarse,
                           text_column_name="text",
                           label_column_name="coarse_label")
    prediction_head_coarse = TextClassificationHead(
        num_labels=len(LABEL_LIST_COARSE),
        task_name="coarse",
        class_weights=None)
    prediction_heads.append(prediction_head_coarse)
if PREDICT == "fine" or PREDICT == "both":
    processor.add_task(name="fine",
                           task_type="classification",
                           label_list=LABEL_LIST_FINE,
                           metric=metrics_fine,
                           text_column_name="text",
                           label_column_name="fine_label")
    prediction_head_fine = TextClassificationHead(
        num_labels=len(LABEL_LIST_FINE),
        task_name="fine",
        class_weights=None)
    prediction_heads.append(prediction_head_fine)
# processor.save("mtl01-model")

data_silo = DataSilo(
    processor=processor,
    batch_size=BATCH_SIZE)

language_model = LanguageModel.load(LANG_MODEL)


def loss_round_robin(tensors, global_step, batch=None):
    if global_step % 2:
        return tensors[0]
    else:
        return tensors[1]


if PREDICT == "both" and DO_ROUND_ROBIN:
    loss_fn = loss_round_robin
else:
    loss_fn = None


model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=prediction_heads,
    embeds_dropout_prob=EMBEDS_DROPOUT_PROB,
    lm_output_types=["per_sequence", "per_sequence"],
    loss_aggregation_fn=loss_fn,
    device=device)


model, optimizer, lr_schedule = initialize_optimizer(
    model=model,
    device=device,
    learning_rate=LEARNING_RATE,
    n_batches=len(data_silo.loaders["train"]),
    n_epochs=MAX_N_EPOCHS)


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=MAX_N_EPOCHS,
    n_gpu=N_GPU,
    lr_schedule=lr_schedule,
    evaluate_every=EVAL_EVERY,
    device=device,
)


logger.info("Starting training")
model = trainer.train()
# model.save("mtl01-model")


inferencer = Inferencer(model=model,
                        processor=processor,
                        batch_size=4, gpu=True,
                        # TODO: how to mix for multihead?
                        task_type="classification"
                        )
basic_texts = [
    {"text": "Some text you want to classify"},
    {"text": "A second sample"},
]


ret = inferencer.inference_from_dicts(basic_texts)
logger.info(f"Result of inference: {ret}")

logger.info(f"Evaluating on training set...")
evaluator = Evaluator(
    data_loader=data_silo.get_data_loader("train"),
    tasks=processor.tasks,
    device=device)

result = evaluator.eval(
    inferencer.model,
    return_preds_and_labels=True)

evaluator.log_results(
    result,
    "Test",
    steps=len(data_silo.get_data_loader("test")))

inferencer.close_multiprocessing_pool()
logger.info("PROCESSING FINISHED")

