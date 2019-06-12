
from opensesame.data_handler.seq_classification import GNADProcessor
from opensesame.models.bert.training import run_model
import argparse
from opensesame.file_utils import read_config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf_file",
                        help="Specify config file",
                        metavar="FILE",
                        default="gnad_config.json")
    cli_args, remaining_argv = parser.parse_known_args()
    args = read_config(cli_args.conf_file)

    metric = "acc"
    processor = GNADProcessor(data_dir=args.data_dir,dev_size=args.dev_size,seed=args.seed)
    output_mode = "classification"


    models = ["bert-base-multilingual-uncased"]


    for model in models:
        args.bert_model = model
        args.mlflow_run_name = "tm " + model
        run_model(args=args, prediction_head="seq_classification", processor=processor, output_mode=output_mode,
                  metric=metric)


if __name__ == "__main__":
    main()
