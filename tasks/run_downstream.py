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

import numpy as np

from opensesame.data_handler.seq_classification import GermEval18coarseProcessor, GermEval18fineProcessor, GNADProcessor
from opensesame.data_handler.ner import ConllProcessor
from opensesame.models.bert.training import run_model
import argparse
from opensesame.file_utils import read_config

def unnestConfig(config, flattened=False):
    """
    This function unnests config files e.g. for doing grid search over the parameter space. If a config parameter is a list
    of values this list is iterated over and a config object without lists is returned to do training. It can handle
    lists at multiple locations
    :param config:
    :type config:
    :param flattened:
    :type flattened:
    :return:
    :rtype:
    """
    def __deep_dict_access(x, keylist):
        """
        Private function to access nested dictionary elements
        """
        val = x
        for key in keylist:
            val = val[key]
        return val

    unnestedConfig = []
    if(flattened):
        nestedKeys = []
        nestedVals = []
        for k, v in config.items():
            if(isinstance(v,list)):
                nestedKeys.append(k)
                nestedVals.append(v)
        if(len(nestedKeys)>0):
            mesh = np.meshgrid(*nestedVals) # get all combinations, each dimension corresponds to one parameter type
            #flatten mesh into shape: [num_parameters, num_combinations]
            for i in range(len(mesh)):
                mesh[i] = mesh[i].flatten()

            # loop over all combinations
            for i in range(len(mesh[0])):
                tempconfig = config.copy()
                for k in range(len(nestedKeys)):
                    tempconfig[nestedKeys[k]] = mesh[k][i] #get ith val of correct param value
                unnestedConfig.append(tempconfig)
        else:
            unnestedConfig = config
    else:
        #TODO work on unflattend config,
        # guess we need to have nestedkeys as a list of single keys. e.g. ["paramters","learning_rate"] and use deep_dict_access()


        unnestedConfig = config
    return unnestedConfig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--conf_file",
                        help="Specify config file",
                        metavar="FILE",
                        default="seq_classification/germEval18_config.json")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file,flattend=True)

    if(args.name == "GermEval18Coarse"):
        processor = GermEval18coarseProcessor(args.data_dir, args.dev_size, args.seed)
    elif (args.name == "GermEval18Fine"):
        processor = GermEval18fineProcessor(args.data_dir, args.dev_size, args.seed)
    elif (args.name == "Conll2003"):
        processor = ConllProcessor(args.data_dir)
    elif (args.name == "GermEval14"):
        processor = ConllProcessor(args.data_dir)
    elif (args.name == "GNAD"):
        processor = GNADProcessor(args.data_dir, args.dev_size, args.seed)
    else:
        raise NotImplementedError

    configList = unnestConfig(args,flattened=True)
    for conf in configList:
        #run_model(args=conf, prediction_head=args.prediction_head, processor=processor, output_mode=args.output_mode,metric=args.metric)
        print(conf)
        muh =1


if __name__ == "__main__":
    main()
