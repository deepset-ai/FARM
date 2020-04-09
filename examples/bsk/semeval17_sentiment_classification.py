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
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings, flatten_list

from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report
import pandas as pd

def doc_classifcation():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="SemEval17_Sentiment", run_name="3e-5 + 4 Epochs")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    n_epochs = 1
    batch_size = 90
    evaluate_every = 300
    lang_model = "bert-base-cased"
    do_lower_case = False
    # or a local path:
    # lang_model = Path("../saved_models/farm-bert-base-cased")
    use_amp = None

    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load GermEval 2018 Data.

    label_list = ["negative", "neutral", "positive"]
    metric = "f1_macro"

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir=Path("sentiment_data/semeval-17"),
                                            train_filename="dev_train.tsv",
                                            test_filename="test.tsv",
                                            dev_filename=None,
                                            dev_split=0.1,
                                            label_list=label_list,
                                            metric=metric,
                                            label_column_name="sentiment",
                                            quote_char='"'
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    #    few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
        num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=5e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        use_amp=use_amp)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("saved_models/bert-english-sentiment-semeval17_1")
    model.save(save_dir)
    processor.save(save_dir)

    model = Inferencer.load(save_dir, gpu=True)
    preds = model.inference_from_file("sentiment_data/semeval-17/test.tsv")
    pred_labels = [instance["label"] for instance in flatten_list([batch["predictions"] for batch in preds])]

    true_labels = pd.read_csv("sentiment_data/semeval-17/test.tsv", sep="\t").sentiment

    evaluation_measures = classification_report(true_labels, pred_labels, output_dict=True)
    average_recall = evaluation_measures["macro avg"]["recall"]
    accuracy = evaluation_measures["accuracy"]
    f1_pn = ((evaluation_measures["positive"]["f1-score"] + evaluation_measures["negative"]["f1-score"]) / 2)
    print("Average Recall:", average_recall)
    print("Accuracy:", accuracy)
    print("Micro-averaged F1 (pos. + neg.):", f1_pn)

    ml_logger.log_metrics({"Average Recall" : average_recall,
                           "Accuracy" : accuracy,
                           "Micro-averaged F1" : f1_pn}, step=0)

if __name__ == "__main__":
    doc_classifcation()

# fmt: on
