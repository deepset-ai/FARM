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
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

import pandas as pd
from sklearn.metrics import classification_report

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="http://mlflow:hYfk-Pkdrt-53Jre-Ps7N@mlflow-research.deepset.ai")
    ml_logger.init_experiment(experiment_name="Fine-tune xlm-roberta-semeval on germeval", run_name="fine-tuned timestamp2")

    model_path = "saved_models/xlm-roberta-large-english-SemEval2017-4a-english-Fine-tune xlm-roberta-semeval on germeval"
    model = Inferencer.load(model_path, batch_size=30, gpu=True)

    dataframe = pd.read_csv("sentiment_data/germeval-17/test_TIMESTAMP2_normalized.tsv", sep="\t")
    true_labels = dataframe["sentiment"]
    texts = [{"text" : text} for text in dataframe["text"]]


    result = model.inference_from_dicts(texts)
    predictions = [instance.get("label") for batch in result for instance in batch.get("predictions")]

    metrics = classification_report(true_labels, predictions, output_dict=True)
    f1_micro = metrics["accuracy"]
    f1_macro = metrics["macro avg"]["f1-score"]
    f1_weighted = metrics["weighted avg"]["f1-score"]

    ml_logger.log_metrics({"Weighted-averaged F1": f1_weighted,
                           "Macro-averaged F1": f1_macro,
                           "Micro-averaged F1": f1_micro}, step=0)
if __name__ == '__main__':
    main()