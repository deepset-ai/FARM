
from opensesame.data_handler.seq_classification import GNADProcessor
from opensesame.models.bert.training import run_seq_classification
import argparse
from opensesame.file_utils import read_config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file)

    #TODO option to write arguments to model_config and log it as artifact in mlflow

    metric = "acc"
    processor = GNADProcessor()
    output_mode = "classification"

    run_seq_classification(args, processor, output_mode, metric)

if __name__ == "__main__":
    main()
