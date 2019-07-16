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
"""BERT finetuning runner for all tasks in specified config files."""

import logging
from farm.experiment import run_experiment
from farm.file_utils import read_config, unnestConfig

logger = logging.getLogger(__name__)
from farm.utils import MLFlowLogger


def main():
    config_files = [
        "tasks/ner/conll2003_de_config.json",
        "tasks/ner/germEval14_config.json",
        "tasks/seq_classification/germEval18Fine_config.json",
        "tasks/seq_classification/germEval18Coarse_config.json",
        "tasks/seq_classification/gnad_config.json",
    ]

    for conf_file in config_files:
        args = read_config(conf_file, flattend=True)
        experiments = unnestConfig(args, flattened=True)
        for args in experiments:
            logger.info(
                "\n***********************************************"
                f"\n************* Experiment: {args.name} ************"
                "\n************************************************"
            )
            ml_logger = MLFlowLogger(tracking_uri=args.mlflow_url)
            ml_logger.init_experiment(
                experiment_name=args.mlflow_experiment,
                run_name=args.mlflow_run_name,
                nested=args.mlflow_nested,
            )
            run_experiment(args)


if __name__ == "__main__":
    main()
