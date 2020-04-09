import logging
import json
from pathlib import Path

from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer, EarlyStopping
from farm.eval import Evaluator
from farm.evaluation.metrics import squad
from farm.infer import Inferencer
from farm.data_handler.utils import write_squad_predictions


def question_answering_crossvalidation():
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

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    # for local logging instead:
    # ml_logger = MLFlowLogger(tracking_uri="logs")
    ml_logger.init_experiment(experiment_name="Covid_QA_XVAL", run_name="Crossvalidation")

    ##########################
    ########## Settings
    ##########################
    xval_folds = 5
    xval_stratified = False # stratified sampling of prediction labels doesn't make sense with question-answering

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 3
    batch_size = 24
    evaluate_every = 100
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False
    use_amp = None

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        train_filename="answers.json",
        dev_filename=None,
        dev_split=0.1,
        test_filename=None,
        data_dir=Path("../data/covid"),
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # Load one silo for each fold in our cross-validation
    silos = DataSiloForCrossVal.make(data_silo, n_splits=xval_folds, stratified=xval_stratified)

    # the following steps should be run for each of the folds of the cross validation, so we put them
    # into a function
    def train_on_split(silo_to_use, n_fold, save_dir):
        logger.info(f"############ Crossvalidation: Fold {n_fold} ############")
        # Create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis
        language_model = LanguageModel.load(lang_model)
        # b) and a prediction head on top that is suited for our task => Text classification
        prediction_head = QuestionAnsweringHead()

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm_output_types=["per_token"],
            device=device)

        # Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=3e-5,
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
            metric="f1", mode="max",  # use the metric from our own metrics function instead of loss
            save_dir=save_dir,  # where to save the best model
            patience=2  # number of evaluations to wait for improvement before terminating the training
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
            early_stopping=earlystopping,)

        # train it
        trainer.train()

        return trainer.model

    # for each fold, run the whole training, earlystopping to get a model, then evaluate the model
    # on the test set of each fold
    # Remember all the results for overall metrics over all predictions of all folds and for averaging
    allresults = []
    all_preds = []
    all_labels = []
    bestfold = None
    best_squad = -1
    save_dir = Path("saved_models/roberta-qa-covid")

    for num_fold, silo in enumerate(silos):
        model = train_on_split(silo, num_fold, save_dir)

        # do eval on test set here (and not in Trainer),
        # so that we can easily store the actual preds and labels for a "global" eval across all folds.
        evaluator_test = Evaluator(
            data_loader=silo.get_data_loader("test"),
            tasks=silo.processor.tasks,
            device=device
        )
        result = evaluator_test.eval(model, return_preds_and_labels=True)
        evaluator_test.log_results(result, "Test", steps=len(silo.get_data_loader("test")), num_fold=num_fold)

        allresults.append(result)
        all_preds.extend(result[0].get("preds"))
        all_labels.extend(result[0].get("labels"))

        # keep track of best fold
        squad_score = result[0]["f1"]
        if squad_score > best_squad:
            best_squad = squad_score
            bestfold = num_fold

    # calculate overall metrics across all folds
    xval_score = squad(preds=all_preds, labels=all_labels)

    logger.info(f"XVAL EM:   {xval_score['EM']}")
    logger.info(f"XVAL f1:   {xval_score['f1']}")
    ml_logger.log_metrics({"XVAL EM": xval_score["EM"]}, 0)
    ml_logger.log_metrics({"XVAL f1": xval_score["f1"]}, 0)
    print(allresults)

if __name__ == "__main__":
    question_answering_crossvalidation()