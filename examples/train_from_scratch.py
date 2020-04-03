import argparse
import logging
from pathlib import Path

from farm.modeling.tokenization import Tokenizer
from farm.data_handler.data_silo import StreamingDataSilo, DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


# Launch this via
# python -m torch.distributed.launch --nproc_per_node=4 train_from_scratch.py

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    return args


def train_from_scratch():
    # We need the local rank argument for DDP
    args = parse_arguments()
    use_amp = None  # using "O2" here allows roughly 30% larger batch_sizes and 45% speed up

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.local_rank in [-1, 0]:
        ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
        ml_logger.init_experiment(experiment_name="train_from_scratch", run_name="debug")

    set_all_seeds(seed=39)
    device, n_gpu = initialize_device_settings(use_cuda=False, local_rank=args.local_rank, use_amp=use_amp)

    save_dir = Path("saved_models/train_from_scratch")
    data_dir = Path("data/lm_finetune_nips")
    train_filename = "train.txt"
    dev_filename = "dev.txt"

    max_seq_len = 64
    batch_size = 5#60
    grad_acc = 1
    learning_rate = 0.0001
    warmup_proportion = 0.01
    n_epochs = 5
    evaluate_every = 15000
    checkpoint_every = 10
    # checkpoint_every = None
    # checkpoint_root_dir = None
    checkpoint_root_dir = Path("checkpoints")
    # checkpoint_root_dir = Path("/opt/ml/checkpoints/training")
    distributed = False

    # Choose enough workers to queue sufficient batches during training.
    # 16 works well on a 4x V100 machine with 16 cores.
    # For a single GPU you will need less.
    data_loader_workers = 4

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load("bert-base-uncased", do_lower_case=True)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a PyTorch Dataset
    processor = BertStyleLMProcessor(
        data_dir=data_dir,
        tokenizer=tokenizer, max_seq_len=max_seq_len,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=None,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and
    #    calculates a few descriptive statistics of our datasets
    stream_data_silo = StreamingDataSilo(processor=processor, batch_size=batch_size, distributed=distributed,
                                         dataloader_workers=data_loader_workers)

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
        schedule_opts={"name": "Constant"},
        n_batches=len(stream_data_silo.get_data_loader("train")),
        n_epochs=n_epochs,
        device=device,
        grad_acc_steps=grad_acc,
        distributed=distributed,
        use_amp=use_amp,
        local_rank=args.local_rank
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
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
        local_rank=args.local_rank,
        checkpoint_every=checkpoint_every,
        checkpoint_root_dir=checkpoint_root_dir,
        checkpoints_to_keep=4,
        use_amp=use_amp,
        log_learning_rate=True
    )
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    train_from_scratch()