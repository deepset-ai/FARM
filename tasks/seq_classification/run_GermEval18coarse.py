# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


from opensesame.data_handler.seq_classification import GermEval18coarseProcessor
from opensesame.models.bert.training import run_model
import argparse
from opensesame.file_utils import read_config



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--conf_file",
                        help="Specify config file",
                        metavar="FILE",
                        default="germEval18_config.json")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file)
    metric = "f1_macro"
    processor = GermEval18coarseProcessor(args.data_dir, args.dev_size, args.seed)
    output_mode = "classification"

    run_model(args=args, prediction_head="seq_classification", processor=processor, output_mode=output_mode,
              metric=metric)


if __name__ == "__main__":
    main()
