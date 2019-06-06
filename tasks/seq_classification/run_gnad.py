
from opensesame.data_handler.seq_classification import GNADProcessor
from opensesame.models.bert.training import run_model
import argparse
from opensesame.file_utils import read_config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file)

    metric = "acc"
    processor = GNADProcessor()
    output_mode = "classification"

    run_model(args=args, prediction_head="seq_classification", processor=processor, output_mode=output_mode,
              metric=metric)

if __name__ == "__main__":
    main()
