# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downstream runner for all experiments in specified config files."""

from pathlib import Path
from farm.experiment import run_experiment, load_experiments
from farm.metrics import register_metrics
import torch
from sklearn.metrics import classification_report

CONFIG_FILES = {
    #"germEval17-B_synchronic" : Path("configs/germeval17_timestamp1_config.json"),
    #"germEval17-B_diachronic" : Path("configs/germeval17_timestamp2_config.json"),
    #"semEval17-4A" : Path("configs/semeval17_config.json"),
    #"semEval17-4A" : Path("configs/semeval17_albert_config.json"),
    #"semEval17-4A_xlmroberta" : Path("configs/semeval17_xlmroberta_config.json")
    "finetune semeval-xlm-roberta on germeval" : Path("configs/germerval_semeval_xlmroberta_config.json")
}

def semeval17_metrics(preds, labels):
    evaluation_measures = classification_report(y_true=labels, y_pred=preds, output_dict=True)
    recall_macro = evaluation_measures["macro avg"]["recall"]
    accuracy = evaluation_measures["accuracy"]
    f1_pn = ((evaluation_measures["positive"]["f1-score"] + evaluation_measures["negative"]["f1-score"]) / 2)
    return {
        "recall_macro" : recall_macro,
        "accuracy" : accuracy,
        "f1_pn" : f1_pn,
    }

def germeval17_metrics(preds, labels):
    evaluation_measures = classification_report(y_true=labels, y_pred=preds, output_dict=True)
    f1_micro = evaluation_measures["accuracy"]
    f1_macro = evaluation_measures["macro avg"]["f1-score"]
    f1_weighted = evaluation_measures["weighted avg"]["f1-score"]
    return {"Weighted-averaged F1": f1_weighted,
            "Macro-averaged F1": f1_macro,
            "Micro-averaged F1": f1_micro}


def main():
    register_metrics("semeval17_metrics", semeval17_metrics)
    register_metrics("germeval17_metrics", germeval17_metrics)

    for i, (conf_name, conf_file) in enumerate(CONFIG_FILES.items()):
        experiments = load_experiments(conf_file)
        for j, experiment in enumerate(experiments):
            mlflow_run_name = f"Fine-tuned timestamp2"
            experiment.logging.mlflow_run_name = mlflow_run_name
            run_experiment(experiment)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
