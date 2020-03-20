import argparse
import torch
import json
import logging
from pathlib import Path

from transformers.tokenization_bert import BertTokenizer

from farm.data_handler.data_silo import StreamingDataSilo, DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    return args


def initialize_device_settings(use_cuda, args, local_rank=-1, use_amp=None):
    if not use_cuda:
        device = torch.device("cpu")
        n_gpu = 0
    elif args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            n_gpu = 0
        else:
            n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    print(
        "device: {} n_gpu: {}, distributed training: {}, automatic mixed precision training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), use_amp
        )
    )
    return device, n_gpu

def train_from_scratch():

    # We need the local rank argument for DDP
    args = parse_arguments()
    use_amp = "O2"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ml_logger = MLFlowLogger(tracking_uri="")
    ml_logger.init_experiment(experiment_name="train_from_scratch", run_name="run")

    set_all_seeds(seed=39)
    #device, n_gpu = initialize_device_settings(use_cuda=True)
    device, n_gpu = initialize_device_settings(use_cuda=True, args=args, use_amp=use_amp)
    evaluate_every = 10000

    save_dir = Path("saved_models/train_from_scratch")
    data_dir = Path("data/lm_finetune_nips")
    train_filename = "train.txt"
    #dev_filename = "dev.txt"

    max_seq_len = 128
    batch_size = 80
    grad_acc = 3
    learning_rate = 0.0001
    warmup_proportion = 0.01
    n_epochs = 1
    vocab_file = "bert-base-uncased-vocab.txt"


    # 1.Create a tokenizer
    tokenizer = BertTokenizer(data_dir/vocab_file, do_lower_case=True)
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 2. Create a DataProcessor that handles all the conversion from raw text into a PyTorch Dataset
    processor = BertStyleLMProcessor(
        data_dir=data_dir,
        tokenizer=tokenizer, max_seq_len=max_seq_len,
        train_filename=train_filename,
        dev_filename=None,
        test_filename=None,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and
    #    calculates a few descriptive statistics of our datasets
    stream_data_silo = StreamingDataSilo(processor=processor, batch_size=batch_size, distributed=True)
    
    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.from_scratch("bert", tokenizer.vocab_size)

    # b) and *two* prediction heads on top that are suited for our task => Language Model finetuning
    lm_prediction_head = BertLMHead(768, tokenizer.vocab_size)
    next_sentence_head = NextSentenceHead([768, 2], task_name="nextsentence")

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[lm_prediction_head, next_sentence_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token", "per_sequence"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=learning_rate,
        schedule_opts={"name": "LinearWarmup", "warmup_proportion": warmup_proportion},
        n_batches=len(stream_data_silo.get_data_loader("train")),
        n_epochs=n_epochs,
        device=device,
        grad_acc_steps=grad_acc,
        distributed=True,
        use_amp=use_amp,
        local_rank=args.local_rank
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    # if args.get("checkpoint_every"):
    #     checkpoint_every = int(args["checkpoint_every"])
    #     checkpoint_root_dir = Path("/opt/ml/checkpoints/training")
    # else:
    checkpoint_every = None
    checkpoint_root_dir = None

    trainer = Trainer.create_or_load_checkpoint(
        model=model,
        optimizer=optimizer,
        data_silo=stream_data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
        grad_acc_steps=grad_acc,
        checkpoint_every=checkpoint_every,
        checkpoint_root_dir=checkpoint_root_dir,
        use_amp=use_amp,
    )
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    train_from_scratch()
