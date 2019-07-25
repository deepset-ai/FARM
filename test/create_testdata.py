import argparse
import json
import logging
import os
import pprint

logger = logging.getLogger(__name__)

def squad_subsample():
    if not os.path.exists("../data/samples/squad20"):
        os.makedirs("../data/samples/squad20")

    with open('../data/squad20/dev-v2.0.json') as json_file:
        data = json.load(json_file)
    ss = data["data"][0]["paragraphs"][:5]
    sample = {}
    sample["data"] = [{"paragraphs": ss}]
    #pprint.pprint(sample)
    # just creating same train and dev files
    with open('../data/samples/squad20/dev-sample.json', 'w') as outfile:
        json.dump(sample, outfile)
    with open('../data/samples/squad20/train-sample.json', 'w') as outfile:
        json.dump(sample, outfile)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help="Which task to create testdata for. squad20, ...")
    args = parser.parse_args()
    if(args.task == "squad20"):
        logger.info("Creating test data for squad20, please make sure the data is already downloaded and in data/squad20")
        squad_subsample()