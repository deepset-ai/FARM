import logging
import json
import torch
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

    #ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    # for local logging instead:
    ml_logger = MLFlowLogger(tracking_uri="logs")
    #ml_logger.init_experiment(experiment_name="QA_X-Validation", run_name="Squad_Roberta_Base")

    ##########################
    ########## Settings
    ##########################
    xval_folds = 5

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 3
    batch_size = 20
    evaluate_every = 0
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = True
    dev_split = 0.0
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
        train_filename="dev-v2.0.json", # "flat_short.json", "200406shortanswers.json", "200406answers.json", "subset.json", "flat_short_shortanswers.json"
        dev_filename=None,
        dev_split=0,
        test_filename=None,
        data_dir=Path("../data/squad20"),
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # Load one silo for each fold in our cross-validation
    silos = DataSiloForCrossVal.make_question_answering(data_silo, n_splits=xval_folds, dev_split=dev_split)

    # the following steps should be run for each of the folds of the cross validation, so we put them
    # into a function
    def train_on_split(silo_to_use, n_fold, save_dir):
        logger.info(f"############ Crossvalidation: Fold {n_fold} ############")

        # fine-tune pre-trained question-answering model
        model = AdaptiveModel.convert_from_transformers(lang_model, device, "question_answering")
        model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

        # or train question-answering models from scratch
        ## Create an AdaptiveModel
        ## a) which consists of a pretrained language model as a basis
        ## language_model = LanguageModel.load(lang_model)
        ## b) and a prediction head on top that is suited for our task => Question-answering
        ## prediction_head = QuestionAnsweringHead()
        ## model = AdaptiveModel(
        ##    language_model=language_model,
        ##    prediction_heads=[prediction_head],
        ##    embeds_dropout_prob=0.1,
        ##    lm_output_types=["per_token"],
        ##    device=device,)


        # Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=5e-6,
            device=device,
            n_batches=len(silo_to_use.loaders["train"]),
            n_epochs=n_epochs,
            use_amp=use_amp)

        # Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
        # Also create an EarlyStopping instance and pass it on to the trainer

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=silo_to_use,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device)

        # train it
        trainer.train()

        return trainer.model

    # for each fold, run the whole training, then evaluate the model on the test set of each fold
    # Remember all the results for overall metrics over all predictions of all folds and for averaging
    allresults = []
    all_preds = []
    all_labels = []
    all_f1 = []
    all_em = []
    bestfold = None
    best_squad = -1
    save_dir = Path("saved_models/roberta-qa-squad")

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
        evaluator_test.log_results(result, "Test", logging=False, steps=len(silo.get_data_loader("test")), num_fold=num_fold)

        allresults.append(result)
        all_preds.extend(result[0].get("preds"))
        all_labels.extend(result[0].get("labels"))
        all_f1.append(result[0]["f1"])
        all_em.append(result[0]["EM"])

        # keep track of best fold
        squad_score = result[0]["f1"]
        if squad_score > best_squad:
            best_squad = squad_score
            bestfold = num_fold

    # Save the per-fold results to json for a separate, more detailed analysis
    # with open("qa_xval.results.json", "wt") as fp:
    #     json.dump(allresults, fp)

        # model.cpu()
        # torch.cuda.empty_cache()

    # calculate overall metrics across all folds
    xval_score = squad(preds=all_preds, labels=all_labels)

    logger.info(f"Single EM-Scores:   {all_em}")
    logger.info(f"Single F1-Scores:   {all_f1}")
    logger.info(f"XVAL EM:   {xval_score['EM']}")
    logger.info(f"XVAL f1:   {xval_score['f1']}")
    ml_logger.log_metrics({"XVAL EM": xval_score["EM"]}, 0)
    ml_logger.log_metrics({"XVAL f1": xval_score["f1"]}, 0)
if __name__ == "__main__":
    question_answering_crossvalidation()