# fmt: off
import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_doc_classification")

##########################
########## Settings
##########################
set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
n_epochs = 1
batch_size = 32

evaluate_every = 500
lang_model = "bert-base-uncased"
do_lower_case = True

# 1.Create a tokenizer
tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=do_lower_case)

# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
# Here we load GermEval 2018 Data.

label_list = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
metric = "acc"

processor = TextClassificationProcessor(tokenizer=tokenizer,
                                        max_seq_len=128,
                                        data_dir="../data/toxic-comments",
                                        label_list=label_list,
                                        label_column_name="label",
                                        metric=metric,
                                        quote_char='"',
                                        multilabel=True,
                                        train_filename="train.tsv",
                                        dev_filename="val.tsv",
                                        test_filename=None,
                                        dev_split=0
                                        )

# 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size)

# 4. Create an AdaptiveModel
# a) which consists of a pretrained language model as a basis
language_model = LanguageModel.load(lang_model)
# b) and a prediction head on top that is suited for our task => Text classification
prediction_head = MultiLabelTextClassificationHead(layer_dims=[768, len(processor.tasks["text_classification"]["label_list"])])

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[prediction_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_sequence"],
    device=device)

# 5. Create an optimizer
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=3e-5,
    warmup_proportion=0.1,
    n_batches=len(data_silo.loaders["train"]),
    n_epochs=n_epochs)

# 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=n_gpu,
    warmup_linear=warmup_linear,
    evaluate_every=evaluate_every,
    device=device)

# 7. Let it grow
model = trainer.train(model)

# 8. Hooray! You have a model. Store it:
save_dir = "../saved_models/bert-german-multi-doc-tutorial"
model.save(save_dir)
processor.save(save_dir)

# 9. Load it & harvest your fruits (Inference)
basic_texts = [
    {"text": "You fucking bastards"},
    {"text": "What a lovely world"},
]
model = Inferencer.load(save_dir)
result = model.inference_from_dicts(dicts=basic_texts)
print(result)


# fmt: on
