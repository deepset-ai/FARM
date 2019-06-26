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

import numpy as np
import logging

from opensesame.data_handler.seq_classification import GermEval18coarseProcessor, GermEval18fineProcessor, GNADProcessor
from opensesame.data_handler.ner import ConllProcessor
from opensesame.models.bert.training import run_model
import argparse
from opensesame.file_utils import read_config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def unnestConfig(config, flattened=False):
    """
    This function creates a list of config files for doing grid search over multiple parameters. If a config parameter is a list
    of values this list is iterated over and a config object without lists is returned to do training. It can handle
    lists at multiple locations
    """
    nestedKeys = []
    nestedVals = []
    if(flattened):
        for k, v in config.items():
            if(isinstance(v,list)):
                nestedKeys.append(k)
                nestedVals.append(v)
    else:
        for gk, gv in config.items():
            for k, v in gv.items():
                if(isinstance(v, list)):
                    if (isinstance(v, list)):
                        nestedKeys.append([gk, k])
                        nestedVals.append(v)
                    elif(isinstance(v,dict)):
                        logger.error("Config too deep!")


    if(len(nestedKeys)>0):
        unnestedConfig = []
        mesh = np.meshgrid(*nestedVals) # get all combinations, each dimension corresponds to one parameter type
        #flatten mesh into shape: [num_parameters, num_combinations]
        mesh = [x.flatten() for x in mesh]

        # loop over all combinations
        for i in range(len(mesh[0])):
            tempconfig = config.copy()
            for j,k in enumerate(nestedKeys):
                if(isinstance(k,str)):
                    tempconfig[k] = mesh[j][i] #get ith val of correct param value and overwrite original config
                elif(len(k) == 2):
                    tempconfig[k[0]][k[1]] = mesh[j][i] #set nested dictionary keys
                else:
                    logger.error("Config too deep!")
            unnestedConfig.append(tempconfig)
    else:
        unnestedConfig = [config]


    return unnestedConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--conf_file",
                        help="Specify config file",
                        metavar="FILE",
                        default="seq_classification/gnad_config.json")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file,flattend=True)
    configList = unnestConfig(args, flattened=True)


    # TODO here args is flat, we want nested anyways!
    if(args.name == "GermEval18Coarse"):
        processor = GermEval18coarseProcessor(args.data_dir, args.dev_size, args.seed)
    elif (args.name == "GermEval18Fine"):
        processor = GermEval18fineProcessor(args.data_dir, args.dev_size, args.seed)
    elif (args.name == "Conll2003"):
        processor = ConllProcessor()
    elif (args.name == "GermEval14"):
        processor = ConllProcessor()
    elif (args.name == "GNAD"):
        processor = GNADProcessor(args.data_dir, args.dev_size, args.seed)
    else:
        raise NotImplementedError

    for args in configList:
        run_model(args=args, prediction_head=args.prediction_head, processor=processor, output_mode=args.output_mode,metric=args.metric)

if __name__ == "__main__":
    main()
