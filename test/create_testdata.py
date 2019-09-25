import argparse
import json
import logging
import os
import pprint

logger = logging.getLogger(__name__)

def squad_subsample():
    if not os.path.exists("samples/qa"):
        os.makedirs("samples/qa")

    with open('../data/squad20/dev-v2.0.json') as json_file:
        data = json.load(json_file)

    ss = data["data"][0]["paragraphs"][:1]
    sample = {}
    sample["data"] = [{"paragraphs": ss}]
    # just creating same train and dev files
    with open('samples/qa/dev-sample.json', 'w') as outfile:
        json.dump(sample, outfile)
    with open('samples/qa/train-sample.json', 'w') as outfile:
        json.dump(sample, outfile)

def germeval14_subsample():
    if not os.path.exists("samples/ner"):
        os.makedirs("samples/ner")

    with open('../data/germeval14/dev.txt') as file:
        data = file.readlines()

    ss = "".join(data[:200])
    with open('samples/ner/train-sample.txt', 'w') as outfile:
        outfile.write(ss)
    with open('samples/ner/dev-sample.txt', 'w') as outfile:
        outfile.write(ss)

def germeval18_subsample():
    if not os.path.exists("samples/doc_class"):
        os.makedirs("samples/doc_class")
    with open('../data/germeval18/test.tsv') as file:
        data = file.readlines()

    ss = "".join(data[:50])
    with open('samples/doc_class/train-sample.tsv', 'w') as outfile:
        outfile.write(ss)
    with open('samples/doc_class/test-sample.tsv', 'w') as outfile:
        outfile.write(ss)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help="Which task to create testdata for: qa, ner, doc_class")
    args = parser.parse_args()
    if(args.task == "qa"):
        logger.info("Creating test data for Question Answering, please make sure the original data is already downloaded and in data/squad20")
        squad_subsample()
    elif(args.task == "ner"):
        logger.info(
            "Creating test data for NER, please make sure the original data is already downloaded and in data/germeval14")
        germeval14_subsample()
    elif(args.task == "doc_class"):
        logger.info(
            "Creating test data for Document Classification, please make sure the original data is already downloaded and in data/germeval18")
        germeval18_subsample()