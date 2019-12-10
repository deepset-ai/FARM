# fmt: off
import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer, EarlyStopping
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.eval import Evaluator
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, f1_score, mean_squared_error, r2_score
from farm.metrics import simple_accuracy, register_metrics
from torch.utils.data import ConcatDataset, Subset
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from farm.data_handler.dataloader import NamedDataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

# ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
# for local logging instead:
ml_logger = MLFlowLogger(tracking_uri="logs")
# ml_logger.init_experiment(experiment_name="Public_FARM", run_name="DocClassification_ES_f1_1")

##########################
########## Settings
##########################
xval_folds = 10
xval_stratified = True

set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
n_epochs = 20
batch_size = 32
evaluate_every = 100
lang_model = "bert-base-german-cased"

# 1.Create a tokenizer
tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=False)

# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
# Here we load GermEval 2018 Data.

# The processor wants to know the possible labels ...
label_list = ["OTHER", "OFFENSE"]

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
        "mcc": mcc,
        "_preds": preds,
        "_labels": labels
    }
register_metrics('mymetrics', mymetrics)
metric = 'mymetrics'

processor = TextClassificationProcessor(tokenizer=tokenizer,
                                        max_seq_len=64,
                                        data_dir="../data/germeval18",
                                        label_list=label_list,
                                        metric=metric,
                                        label_column_name="coarse_label"
                                        )

# 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size)

# For performing cross validation, we really want to combine all the instances from all
# the sets or just some of the sets, then create a different data silo instance for each fold.
# Here, we combine the instances from the train and dev sets to perform xcross validation,
# then create a different data silo instance with train, dev and test sets for each fold
# We use our own DataSiloTmp class to just represent the subsets for train/dev/test for each fold
# as we need it
class DataSilo4Xval:
    def __init__(self, origsilo, trainset, devset, testset):
        self.tensor_names = origsilo.tensor_names
        self.data = {"train":trainset, "dev":devset, "test":testset}
        self.processor = origsilo.processor
        self.batch_size = origsilo.batch_size
        # should not be necessary, xval makes no sense with huge data
        # sampler_train = DistributedSampler(self.data["train"])
        sampler_train = RandomSampler(trainset)

        self.data_loader_train = NamedDataLoader(
            dataset=trainset,
            sampler=sampler_train,
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.data_loader_dev = NamedDataLoader(
            dataset=devset,
            sampler=SequentialSampler(devset),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.data_loader_test = NamedDataLoader(
            dataset=testset,
            sampler=SequentialSampler(testset),
            batch_size=self.batch_size,
            tensor_names=self.tensor_names,
        )
        self.loaders = {
            "train": self.data_loader_train,
            "dev": self.data_loader_dev,
            "test": self.data_loader_test,
        }

    def get_data_loader(self, which):
        return self.loaders[which]

    @staticmethod
    def make(datasilo, sets=["train", "dev", "test"], n_splits=5, stratified=True,
             shuffle=True, random_state=None, dev_split=0.2):
        """
        Create number of folds data-silo-like objects which can be used for training from the
        original data silo passed on.
        :param datasilo: the data silo that contains the original data
        :param sets: which sets to use to create the xval folds
        :param n_splits: number of folds to create
        :param stratified: if class stratificiation should be done
        :param shuffle: shuffle each class' samples before splitting
        :param random_state: random state for shuffling
        :param dev_split: size of the dev set for a fold, held out from the training set
        """
        setstoconcat = [datasilo.data[setname] for setname in sets]
        ds_all = ConcatDataset(setstoconcat)
        idxs = list(range(len(ds_all)))
        if stratified:
            # get all the labels for stratification
            ytensors = [t[3][0] for t in ds_all]
            Y = torch.stack(ytensors)
            xval = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            xval_split = xval.split(idxs,Y)
        else:
            xval = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            xval_split = xval.split(idxs)
        # for each fold create a DataSilo4Xval instance, where the training set is further
        # divided into actual train and dev set
        silos = []
        for train_idx, test_idx in xval_split:
            n_dev = int(dev_split * len(train_idx))
            n_actual_train = len(train_idx) - n_dev
            # TODO: this split into actual train and test set could/should also be stratified, for now
            # we just do this by taking the first/last indices from the train set (which should be
            # shuffled by default)
            actual_train_idx = train_idx[:n_actual_train]
            dev_idx = train_idx[n_actual_train:]
            # create the actual datasets
            ds_train = Subset(ds_all, actual_train_idx)
            ds_dev = Subset(ds_all, dev_idx)
            ds_test = Subset(ds_all, test_idx)
            silos.append(DataSilo4Xval(datasilo, ds_train, ds_dev, ds_test))
        return silos

silos = DataSilo4Xval.make(data_silo, n_splits=3)

# the following steps should be run for each of the folds of the cross validation, so we put them
# into a function
def train_on_split(silo_to_use, foldnr):
    # Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(
        layer_dims=[768, len(processor.tasks["text_classification"]["label_list"])],
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.2,
        lm_output_types=["per_sequence"],
        device=device)

    #  Create an optimizer
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=0.5e-5,
        warmup_proportion=0.1,
        n_batches=len(silo_to_use.loaders["train"]),   # TODO
        n_epochs=n_epochs)

    # Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    # Also create an EarlyStopping instance and pass it on to the trainer

    # An early stopping instance can be used to save the model that performs best on the dev set
    # according to some metric and stop training when no improvement is happening for some iterations.
    # NOTE: if we would use a different save directory for each fold, we could afterwards use a the
    # nfolds best models in an ensemble!
    save_dir = "saved_models/bert-german-doc-tutorial-es-{}".format(foldnr)
    earlystopping = EarlyStopping(
        metric="f1_offense", mode="max",   # use the metric from our own metrics function instead of loss
        # metric="f1_macro", mode="max",  # use f1_macro from the dev evaluator of the trainer
        # metric="loss", mode="min",   # use loss from the dev evaluator of the trainer
        save_dir=save_dir,  # where to save the best model
        patience=5    # number of evaluations to wait for improvement before terminating the training
    )

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=silo_to_use,
        epochs=n_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device,
        early_stopping=earlystopping)

    # train it
    model = trainer.train(model)

    # Since we used early stopping, restore the best model from there.
    lm_name = model.language_model.name
    model = AdaptiveModel.load(earlystopping.save_dir, trainer.device, lm_name=lm_name)
    model.connect_heads_with_processor(silo_to_use.processor.tasks, require_labels=True)
    return model


# for each fold, run the whole training, earlystopping to get a model, then evaluate the model
# on the test set of each fold
# Remember all the results for overall metrics over all predictions of all folds and for averaging
allresults = []
bestfold = None
bestf1_offense = -1
for foldnr, silo in enumerate(silos):
    model = train_on_split(silo, foldnr)
    # make an evaluator for the evaluation on the test set
    evaluator_test = Evaluator(
        data_loader=silo.get_data_loader("test"),
        tasks=silo.processor.tasks,
        device=device
    )
    result = evaluator_test.eval(model)
    allresults.append(result)
    # get the f1_offense metric
    f1_offense = result[0]["f1_offense"]
    if f1_offense > bestf1_offense:
        bestf1_offense = f1_offense
        bestfold = foldnr

# Save the per-fold results to json for a separate, more detailed analysis
import json
with open("doc_classification_xval.results.json", "wt") as fp:
    json.dump(allresults, fp)

# Combine the per-fold labels/predictions so we can calculate overall metrics
all_preds = []
all_labels = []
for result in allresults:
    all_preds.extend(result[0].get("_preds"))
    all_labels.extend(result[0].get("_labels"))

xval_f1_micro = f1_score(all_labels, all_preds, labels=label_list, average="micro")
xval_f1_macro = f1_score(all_labels, all_preds, labels=label_list, average="macro")
xval_f1_offense = f1_score(all_labels, all_preds, labels=label_list, pos_label="OFFENSE")
xval_f1_other = f1_score(all_labels, all_preds, labels=label_list, pos_label="OTHER")
xval_mcc = matthews_corrcoef(all_labels, all_preds)

# TODO: use logger
print("XVAL F1 MICRO:   ", xval_f1_micro)
print("XVAL F1 MACRO:   ", xval_f1_macro)
print("XVAL F1 OFFENSE: ", xval_f1_offense)
print("XVAL F1 OTHER:   ", xval_f1_other)
print("XVAL MCC:        ", xval_mcc)

# Just for illustration, use the best model from the best xval val for evaluation on
# the original (still unseen) test set.
evaluator_origtest = Evaluator(
    data_loader=data_silo.get_data_loader("test"),
    tasks=data_silo.processor.tasks,
    device=device
)
# restore model from the best fold
lm_name = model.language_model.name
save_dir = "saved_models/bert-german-doc-tutorial-es-{}".format(bestfold)
model = AdaptiveModel.load(save_dir, device, lm_name=lm_name)
model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

result = evaluator_origtest.eval(model)
print("TEST F1 MICRO:   ", result[0]["f1_micro"])
print("TEST F1 MACRO:   ", result[0]["f1_macro"])
print("TEST F1 OFFENSE: ", result[0]["f1_offense"])
print("TEST F1 OTHER:   ", result[0]["f1_other"])
print("TEST MCC:        ", result[0]["mcc"])

# fmt: on
