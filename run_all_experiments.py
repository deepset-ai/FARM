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

import logging
from farm.experiment import run_experiment, load_experiments
from farm.utils import MLFlowLogger

logger = logging.getLogger(__name__)


def main():
    config_files = [
        "experiments/ner/conll2003_de_config.json",
        "experiments/ner/germEval14_config.json",
        "experiments/text_classification/germEval18Fine_config.json",
        "experiments/text_classification/germEval18Coarse_config.json",
        "experiments/text_classification/gnad_config.json",
        "experiments/qa/squad20_config.json",
    ]

    for conf_file in config_files:
        experiments = load_experiments(conf_file)
        for args in experiments:
            logger.info(
                "\n***********************************************"
                f"\n************* Experiment: {args.task.name} ************"
                "\n************************************************"
            )
            ml_logger = MLFlowLogger(tracking_uri=args.logging.mlflow_url)
            ml_logger.init_experiment(
                experiment_name=args.logging.mlflow_experiment,
                run_name=args.logging.mlflow_run_name,
                nested=args.logging.mlflow_nested,
            )
            run_experiment(args)


if __name__ == "__main__":
    main()
