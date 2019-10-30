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

from farm.experiment import run_experiment, load_experiments


def main():
    config_files = [
        "experiments/ner/conll2003_de_config.json",
        "experiments/ner/conll2003_en_config.json",
        "experiments/ner/germEval14_config.json",
        "experiments/text_classification/germEval18Fine_config.json",
        "experiments/text_classification/germEval18Coarse_config.json",
        "experiments/text_classification/gnad_config.json",
        "experiments/text_classification/cola_config.json",
        "experiments/qa/squad20_config.json",
    ]

    for conf_file in config_files:
        experiments = load_experiments(conf_file)
        for experiment in experiments:
            run_experiment(experiment)

if __name__ == "__main__":
    main()
