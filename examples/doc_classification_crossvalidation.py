# fmt: off
import logging
import numbers
import json
import mlflow
import statistics
from pathlib import Path
from collections import defaultdict
import torch

from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
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


def doc_classification_crossvalidation():
    ##########################
    ########## Logging
    ##########################
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # reduce verbosity from transformers library
    logging.getLogger('transformers').setLevel(logging.WARNING)

    # ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    # for local logging instead:
    ml_logger = MLFlowLogger(tracking_uri="logs")
    # ml_logger.init_experiment(experiment_name="Public_FARM", run_name="DocClassification_ES_f1_1")

    ##########################
    ########## Settings
    ##########################
    xval_folds = 5
    xval_stratification = True

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 20
    batch_size = 32
    evaluate_every = 100
    dev_split = 0.1
    # For xval the dev_stratification parameter must not be None: with None, the devset cannot be created
    # using the default method of only splitting by the available chunks as initial train set for each fold
    # is just a single chunk!
    dev_stratification = True
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
        f1micro = f1_score(y_true=labels, y_pred=preds, average="micro")
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
                                            dev_split=dev_split,
                                            dev_stratification=dev_stratification,
                                            label_column_name="coarse_label"
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # Load one silo for each fold in our cross-validation
    silos = DataSiloForCrossVal.make(data_silo,
                                     sets=["train", "dev"],
                                     n_splits=xval_folds,
                                     stratification=xval_stratification)

    # the following steps should be run for each of the folds of the cross validation, so we put them
    # into a function
    def train_on_split(silo_to_use, n_fold, save_dir):
        logger.info(f"############ Crossvalidation: Fold {n_fold} of {xval_folds} ############")
        logger.info(f"Fold training   samples: {len(silo_to_use.data['train'])}")
        logger.info(f"Fold dev        samples: {len(silo_to_use.data['dev'])}")
        logger.info(f"Fold testing    samples: {len(silo_to_use.data['test'])}")
        logger.info( "Total number of samples: "
                    f"{len(silo_to_use.data['train'])+len(silo_to_use.data['dev'])+len(silo_to_use.data['test'])}")

        # Create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis
        language_model = LanguageModel.load(lang_model)
        # b) and a prediction head on top that is suited for our task => Text classification
        prediction_head = TextClassificationHead(
            class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
            num_labels=len(label_list))

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.2,
            lm_output_types=["per_sequence"],
            device=device)

        # Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=0.5e-5,
            device=device,
            n_batches=len(silo_to_use.loaders["train"]),
            n_epochs=n_epochs,
            use_amp=use_amp)

        # Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
        # Also create an EarlyStopping instance and pass it on to the trainer

        # An early stopping instance can be used to save the model that performs best on the dev set
        # according to some metric and stop training when no improvement is happening for some iterations.
        # NOTE: Using a different save directory for each fold, allows us afterwards to use the
        # nfolds best models in an ensemble!
        save_dir = Path(str(save_dir) + f"-{n_fold}")
        earlystopping = EarlyStopping(
            metric="f1_offense", mode="max",   # use the metric from our own metrics function instead of loss
            save_dir=save_dir,  # where to save the best model
            patience=5    # number of evaluations to wait for improvement before terminating the training
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=silo_to_use,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device,
            early_stopping=earlystopping,
            evaluator_test=False)

        # train it
        trainer.train()

        return trainer.model

    # for each fold, run the whole training, earlystopping to get a model, then evaluate the model
    # on the test set of each fold

    # remember all individual evaluation results
    allresults = []
    bestfold = None
    bestf1_offense = -1
    save_dir = Path("saved_models/bert-german-doc-tutorial-es")
    for num_fold, silo in enumerate(silos):
        mlflow.start_run(run_name=f"fold-{num_fold + 1}-of-{len(silos)}", nested=True)
        model = train_on_split(silo, num_fold, save_dir)

        # do eval on test set here (and not in Trainer),
        #  so that we can easily store the actual preds and labels for a "global" eval across all folds.
        evaluator_test = Evaluator(
            data_loader=silo.get_data_loader("test"),
            tasks=silo.processor.tasks,
            device=device
        )
        result = evaluator_test.eval(model, return_preds_and_labels=True)
        evaluator_test.log_results(result, "Test", steps=len(silo.get_data_loader("test")), num_fold=num_fold)

        allresults.append(result)

        # keep track of best fold
        f1_offense = result[0]["f1_offense"]
        if f1_offense > bestf1_offense:
            bestf1_offense = f1_offense
            bestfold = num_fold
        mlflow.end_run()
        # emtpy cache to avoid memory leak and cuda OOM across multiple folds
        model.cpu()
        torch.cuda.empty_cache()

    # Save the per-fold results to json for a separate, more detailed analysis
    with open("doc_classification_xval.results.json", "wt") as fp:
        json.dump(allresults, fp)

    # log the best fold metric and fold
    logger.info(f"Best fold f1_offense: {bestf1_offense} in fold {bestfold}")

    # calculate overall metrics across all folds: we only have one head so we do this only for the first head
    # information in each of the per-fold results

    # First create a dict where for each metric, we have a list of values from each fold
    xval_metric_lists_head0 = defaultdict(list)
    for results in allresults:
        head0results = results[0]
        for name in head0results.keys():
            if name not in ["preds", "labels"] and not name.startswith("_") and \
                    isinstance(head0results[name], numbers.Number):
                xval_metric_lists_head0[name].append(head0results[name])
    # Now calculate the mean and stdev for each metric, also copy over the task name
    xval_metric = {}
    xval_metric["task_name"] = allresults[0][0].get("task_name", "UNKNOWN TASKNAME")
    for name in xval_metric_lists_head0.keys():
        values = xval_metric_lists_head0[name]
        vmean = statistics.mean(values)
        vstdev = statistics.stdev(values)
        xval_metric[name+"_mean"] = vmean
        xval_metric[name+"_stdev"] = vstdev

    logger.info(f"XVAL Accuracy:   mean {xval_metric['acc_mean']} stdev {xval_metric['acc_stdev']}")
    logger.info(f"XVAL F1 MICRO:   mean {xval_metric['f1_micro_mean']} stdev {xval_metric['f1_micro_stdev']}")
    logger.info(f"XVAL F1 MACRO:   mean {xval_metric['f1_macro_mean']} stdev {xval_metric['f1_macro_stdev']}")
    logger.info(f"XVAL F1 OFFENSE: mean {xval_metric['f1_offense_mean']} stdev {xval_metric['f1_offense_stdev']}")
    logger.info(f"XVAL F1 OTHER:   mean {xval_metric['f1_other_mean']} stdev {xval_metric['f1_other_stdev']}")
    logger.info(f"XVAL MCC:        mean {xval_metric['mcc_mean']} stdev {xval_metric['mcc_stdev']}")

    # -----------------------------------------------------
    # Just for illustration, use the best model from the best xval val for evaluation on
    # the original (still unseen) test set.
    logger.info("###### Final Eval on hold out test set using best model #####")
    evaluator_origtest = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    # restore model from the best fold
    lm_name = model.language_model.name
    save_dir = Path(f"saved_models/bert-german-doc-tutorial-es-{bestfold}")
    model = AdaptiveModel.load(save_dir, device, lm_name=lm_name)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    result = evaluator_origtest.eval(model)
    logger.info(f"TEST Accuracy:   {result[0]['acc']}")
    logger.info(f"TEST F1 MICRO:   {result[0]['f1_micro']}")
    logger.info(f"TEST F1 MACRO:   {result[0]['f1_macro']}")
    logger.info(f"TEST F1 OFFENSE: {result[0]['f1_offense']}")
    logger.info(f"TEST F1 OTHER:   {result[0]['f1_other']}")
    logger.info(f"TEST MCC:        {result[0]['mcc']}")


if __name__ == "__main__":
    doc_classification_crossvalidation()

# fmt: on
