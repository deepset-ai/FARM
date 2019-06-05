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


from opensesame.data_handler.seq_classification import GermEval18SentimentProcessor
from opensesame.models.bert.training import run_seq_classification
import argparse
from opensesame.file_utils import read_config



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file)

    metric = "acc"
    processor = GermEval18SentimentProcessor(args.data_dir, args.dev_size, args.seed)
    output_mode = "classification"

    run_seq_classification(args, processor, output_mode, metric)


if __name__ == "__main__":
    main()
