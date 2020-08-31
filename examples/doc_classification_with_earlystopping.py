# fmt: off
import logging
from pathlib import Path

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
from sklearn.metrics import f1_score
from farm.evaluation.metrics import simple_accuracy, register_metrics

def doc_classification_with_earlystopping():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    # for local logging instead:
    # ml_logger = MLFlowLogger(tracking_uri="logs")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="DocClassification_ES_f1_1")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    use_amp = None
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 20
    batch_size = 32
    evaluate_every = 100
    lang_model = "bert-base-german-cased"
    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load GermEval 2018 Data automaticaly if it is not available.
    # GermEval 2018 only has train.tsv and test.tsv dataset - no dev.tsv

    # The processor wants to know the possible labels ...
    label_list = ["OTHER", "OFFENSE"]

    # The evaluation on the dev-set can be done with one of the predefined metrics or with a
    # metric defined as a function from (preds, labels) to a dict that contains all the actual
    # metrics values. The function must get registered under a string name and the string name must
    # be used.
    def mymetrics(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1other = f1_score(y_true=labels, y_pred=preds, pos_label="OTHER")
        f1offense = f1_score(y_true=labels, y_pred=preds, pos_label="OFFENSE")
        f1macro = f1_score(y_true=labels, y_pred=preds, average="macro")
        f1micro = f1_score(y_true=labels, y_pred=preds, average="micro")
        return {"acc": acc, "f1_other": f1other, "f1_offense": f1offense, "f1_macro": f1macro, "f1_micro": f1micro}
    register_metrics('mymetrics', mymetrics)
    metric = 'mymetrics'

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=64,
                                            data_dir=Path("../data/germeval18"),
                                            label_list=label_list,
                                            metric=metric,
                                            label_column_name="coarse_label"
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(num_labels=len(label_list),
                                             class_weights=data_silo.calculate_class_weights(task_name="text_classification"))



    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.2,
        lm_output_types=["per_sequence"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=0.5e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        use_amp=use_amp)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    # Also create an EarlyStopping instance and pass it on to the trainer

    # An early stopping instance can be used to save the model that performs best on the dev set
    # according to some metric and stop training when no improvement is happening for some iterations.
    earlystopping = EarlyStopping(
        metric="f1_offense", mode="max",   # use the metric from our own metrics function instead of loss
        # metric="f1_macro", mode="max",  # use f1_macro from the dev evaluator of the trainer
        # metric="loss", mode="min",   # use loss from the dev evaluator of the trainer
        save_dir=Path("saved_models/bert-german-doc-tutorial-es"),  # where to save the best model
        patience=5    # number of evaluations to wait for improvement before terminating the training
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
        early_stopping=earlystopping)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model.
    # NOTE: if early stopping is used, the best model has been stored already in the directory
    # defined with the EarlyStopping instance
    # The model we have at this moment is the model from the last training epoch that was carried
    # out before early stopping terminated the training
    save_dir = Path("saved_models/bert-german-doc-tutorial")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]

    # Load from the final epoch directory and apply
    print("LOADING INFERENCER FROM FINAL MODEL DURING TRAINING")
    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    print(result)
    model.close_multiprocessing_pool()

    # Load from saved best model
    print("LOADING INFERENCER FROM BEST MODEL DURING TRAINING")
    model = Inferencer.load(earlystopping.save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    print("APPLICATION ON BEST MODEL")
    print(result)
    model.close_multiprocessing_pool()


if __name__ == "__main__":
    doc_classification_with_earlystopping()

# fmt: on
