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
"""BERT finetuning runner."""

import argparse
import logging

from farm.data_handler.ner import ConllProcessor
from farm.data_handler.seq_classification import (
    GermEval18coarseProcessor,
    GermEval18fineProcessor,
    GNADProcessor,
)
from farm.modeling.bert.training import run_model

from farm.file_utils import read_config, unnestConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf_file",
        help="Specify config file",
        metavar="FILE",
        default="seq_classification/gnad_config.json",
    )
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file, flattend=True)
    configList = unnestConfig(args, flattened=True)

    # TODO here args is flat, we want nested anyways!
    if args.name == "GermEval18Coarse":
        processor = GermEval18coarseProcessor(args.data_dir, args.dev_size, args.seed)
    elif args.name == "GermEval18Fine":
        processor = GermEval18fineProcessor(args.data_dir, args.dev_size, args.seed)
    elif args.name == "Conll2003":
        processor = ConllProcessor()
    elif args.name == "GermEval14":
        processor = ConllProcessor()
    elif args.name == "GNAD":
        processor = GNADProcessor(args.data_dir, args.dev_size, args.seed)
    else:
        raise NotImplementedError

    for args in configList:
        run_model(
            args=args,
            prediction_head=args.prediction_head,
            processor=processor,
            output_mode=args.output_mode,
            metric=args.metric,
        )


if __name__ == "__main__":
    main()
