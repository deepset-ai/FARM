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

from opensesame.data_handler.seq_classification import GermEval18coarseProcessor, GermEval18fineProcessor, GNADProcessor
from opensesame.data_handler.ner import ConllProcessor
from opensesame.modeling.bert.training import run_model
from opensesame.file_utils import read_config, unnestConfig


logger = logging.getLogger(__name__)


def main():
    config_files = ["ner/conll2003_de_config.json",
                    "ner/germEval14_config.json",
                    "seq_classification/germEval18_config.json",
                    "seq_classification/gnad_config.json"]


    for conf_file in config_files:
        args = read_config(conf_file,flattend=True)
        configList = unnestConfig(args, flattened=True)
        token_level = False
        if(args.name == "GermEval18Coarse"):
            processor = GermEval18coarseProcessor(args.data_dir, args.dev_size, args.seed)
        elif (args.name == "GermEval18Fine"):
            processor = GermEval18fineProcessor(args.data_dir, args.dev_size, args.seed)
        elif (args.name == "Conll2003"):
            processor = ConllProcessor()
            token_level = True
        elif (args.name == "GermEval14"):
            processor = ConllProcessor()
            token_level = True
        elif (args.name == "GNAD"):
            processor = GNADProcessor(args.data_dir, args.dev_size, args.seed)
        else:
            raise NotImplementedError

        for args in configList:
            run_model(args=args, prediction_head=args.prediction_head, processor=processor, output_mode=args.output_mode,metric=args.metric, token_level=token_level)






if __name__ == "__main__":
    main()
