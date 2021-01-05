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
from farm.train import Trainer
from farm.eval import Evaluator
from farm.evaluation.metrics import squad


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
    save_per_fold_results = False # unsupported for now
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)

    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False

    n_epochs = 2
    batch_size = 80
    learning_rate = 3e-5


    data_dir = Path("../data/covidqa")
    filename = "COVID-QA.json"
    xval_folds = 5
    dev_split = 0
    evaluate_every = 0
    no_ans_boost = -100 # use large negative values to disable giving "no answer" option
    accuracy_at = 3 # accuracy at n is useful for answers inside long documents
    use_amp = None

    ##########################
    ########## k fold Cross validation
    ##########################

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=["start_token", "end_token"],
        metric="squad",
        train_filename=filename,
        dev_filename=None,
        dev_split=dev_split,
        test_filename=None,
        data_dir=data_dir,
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # Load one silo for each fold in our cross-validation
    silos = DataSiloForCrossVal.make(data_silo, n_splits=xval_folds)

    # the following steps should be run for each of the folds of the cross validation, so we put them
    # into a function
    def train_on_split(silo_to_use, n_fold):
        logger.info(f"############ Crossvalidation: Fold {n_fold} ############")

        # fine-tune pre-trained question-answering model
        model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="question_answering")
        model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)
        # If positive, thjs will boost "No Answer" as prediction.
        # If negative, this will prevent the model from giving "No Answer" as prediction.
        model.prediction_heads[0].no_ans_boost = no_ans_boost
        # Number of predictions the model will make per Question.
        # The multiple predictions are used for evaluating top n recall.
        model.prediction_heads[0].n_best = accuracy_at

        # # or train question-answering models from scratch
        # # Create an AdaptiveModel
        # # a) which consists of a pretrained language model as a basis
        # language_model = LanguageModel.load(lang_model)
        # # b) and a prediction head on top that is suited for our task => Question-answering
        # prediction_head = QuestionAnsweringHead(no_ans_boost=no_ans_boost, n_best=accuracy_at)
        # model = AdaptiveModel(
        #    language_model=language_model,
        #    prediction_heads=[prediction_head],
        #    embeds_dropout_prob=0.1,
        #    lm_output_types=["per_token"],
        #    device=device,)


        # Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=learning_rate,
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
            device=device,
            evaluator_test=False)

        # train it
        trainer.train()

        return trainer.model

    # for each fold, run the whole training, then evaluate the model on the test set of each fold
    # Remember all the results for overall metrics over all predictions of all folds and for averaging
    all_results = []
    all_preds = []
    all_labels = []
    all_f1 = []
    all_em = []
    all_topnaccuracy = []

    for num_fold, silo in enumerate(silos):
        model = train_on_split(silo, num_fold)

        # do eval on test set here (and not in Trainer),
        # so that we can easily store the actual preds and labels for a "global" eval across all folds.
        evaluator_test = Evaluator(
            data_loader=silo.get_data_loader("test"),
            tasks=silo.processor.tasks,
            device=device
        )
        result = evaluator_test.eval(model, return_preds_and_labels=True)
        evaluator_test.log_results(result, "Test", logging=False, steps=len(silo.get_data_loader("test")), num_fold=num_fold)

        all_results.append(result)
        all_preds.extend(result[0].get("preds"))
        all_labels.extend(result[0].get("labels"))
        all_f1.append(result[0]["f1"])
        all_em.append(result[0]["EM"])
        all_topnaccuracy.append(result[0]["top_n_accuracy"])

        # emtpy cache to avoid memory leak and cuda OOM across multiple folds
        model.cpu()
        torch.cuda.empty_cache()

    # Save the per-fold results to json for a separate, more detailed analysis
    # TODO currently not supported - adjust to QAPred and QACandidate objects
    # if save_per_fold_results:
    #     def convert_numpy_dtype(obj):
    #         if type(obj).__module__ == "numpy":
    #             return obj.item()
    #
    #         raise TypeError("Unknown type:", type(obj))
    #
    #     with open("qa_xval.results.json", "wt") as fp:
    #          json.dump(all_results, fp, default=convert_numpy_dtype)

    # calculate overall metrics across all folds
    xval_score = squad(preds=all_preds, labels=all_labels)


    logger.info(f"Single EM-Scores:   {all_em}")
    logger.info(f"Single F1-Scores:   {all_f1}")
    logger.info(f"Single top_{accuracy_at}_accuracy Scores:   {all_topnaccuracy}")
    logger.info(f"XVAL EM:   {xval_score['EM']}")
    logger.info(f"XVAL f1:   {xval_score['f1']}")
    logger.info(f"XVAL top_{accuracy_at}_accuracy:   {xval_score['top_n_accuracy']}")
    ml_logger.log_metrics({"XVAL EM": xval_score["EM"]}, 0)
    ml_logger.log_metrics({"XVAL f1": xval_score["f1"]}, 0)
    ml_logger.log_metrics({f"XVAL top_{accuracy_at}_accuracy": xval_score["top_n_accuracy"]}, 0)

if __name__ == "__main__":
    question_answering_crossvalidation()
