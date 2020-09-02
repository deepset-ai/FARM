# fmt: off
import logging
import json
from pathlib import Path
import torch

from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal, DataSiloForNestedCrossVal
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer, EarlyStopping
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.eval import Evaluator
from sklearn.metrics import matthews_corrcoef, f1_score
from farm.evaluation.metrics import simple_accuracy, register_metrics

##########################
########## Logging
##########################
logger = logging.getLogger(__name__)
#logging.basicConfig(
#    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#    datefmt="%m/%d/%Y %H:%M:%S",
#    level=logging.INFO)
## reduce verbosity from transformers library
#logging.getLogger('transformers').setLevel(logging.WARNING)

# ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
# for local logging instead:
#ml_logger = MLFlowLogger(tracking_uri="logs")
# ml_logger.init_experiment(experiment_name="Public_FARM", run_name="DocClassification_ES_f1_1")

##########################
########## Settings
##########################
xval_folds = 5
xval_stratified = True

set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
n_epochs = 20
batch_size = 32
evaluate_every = 100
lang_model = "bert-base-german-cased"
do_lower_case = False
use_amp = None

# 1.Create a tokenizer
tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=do_lower_case)

# The evaluation on the dev-set can be done with one of the predefined metrics or with a
# metric defined as a function from (preds, labels) to a dict that contains all the actual
# metrics values. The function must get registered under a string name and the string name must
# be used.
# For xval, we also store the actual predictions and labels in each result so we can
# calculate overall metrics over all folds later
def mymetrics(preds, labels):
    acc = simple_accuracy(preds, labels).get("acc")
    f1other = f1_score(y_true=labels, y_pred=preds, pos_label="OTHER")
    f1offense = f1_score(y_true=labels, y_pred=preds, pos_label="OFFENSE")
    f1macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1micro = f1_score(y_true=labels, y_pred=preds, average="macro")
    mcc = matthews_corrcoef(labels, preds)
    return {
        "acc": acc,
        "f1_other": f1other,
        "f1_offense": f1offense,
        "f1_macro": f1macro,
        "f1_micro": f1micro,
        "mcc": mcc
    }
register_metrics('mymetrics', mymetrics)
metric = 'mymetrics'

# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
# Here we load GermEval 2018 Data automaticaly if it is not available.
# GermEval 2018 only has train.tsv and test.tsv dataset - no dev.tsv

# The processor wants to know the possible labels ...
label_list = ["OTHER", "OFFENSE"]
processor = TextClassificationProcessor(tokenizer=tokenizer,
                                        max_seq_len=64,
                                        data_dir=Path("../data/germeval18"),
                                        label_list=label_list,
                                        metric=metric,
                                        dev_split=0.0,
                                        label_column_name="coarse_label",
                                        test_filename=None,
                                        )

# 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size)

# Load one silo for each fold in our cross-validation
silos = DataSiloForNestedCrossVal.make(
    data_silo,
    n_splits=xval_folds,
    sets=["train"],
    )

for i, silo in enumerate(silos):
    print('>>>', i)
    data_loader_train = silo.get_data_loader("train")
    data_loader_dev = silo.get_data_loader("dev")
    data_loader_test = silo.get_data_loader("test")
    print(len(data_loader_train.dataset.indices))
    print(len(data_loader_dev.dataset.indices))
    print(len(data_loader_test.dataset.indices))

print('#######################')
data_loader_train = data_silo.get_data_loader("train")
data_loader_dev = data_silo.get_data_loader("dev")
data_loader_test = data_silo.get_data_loader("test")
print(len(data_loader_train.dataset.indices))
print(len(data_loader_dev.dataset.indices))
print(len(data_loader_test.dataset.indices))

print('done')
