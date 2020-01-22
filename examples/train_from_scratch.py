# fmt: off
import logging
from pathlib import Path

from transformers.tokenization_bert import BertTokenizer

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


def train_from_scratch():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ml_logger = MLFlowLogger(tracking_uri="")
    ml_logger.init_experiment(experiment_name="from_scratch", run_name="debug")

    #########################
    ######## Settings
    ########################
    set_all_seeds(seed=39)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    evaluate_every = 5000
    vocab_size = 30522
    # dev_filename = None
    save_dir = Path("saved_models/train_from_scratch")

    n_epochs = 10
    learning_rate = 1e-4
    warmup_proportion = 0.05
    batch_size = 16  # (probably only possible via gradient accumulation steps)
    max_seq_len = 64

    # 1.Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = BertStyleLMProcessor(
        data_dir=Path("data/lm_finetune_nips"),
        tokenizer=tokenizer, max_seq_len=max_seq_len,
        train_filename="train.txt",
        dev_split=2000 / 8_000_000,
        dev_filename=None,
        test_filename=None,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and
    #    calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.from_scratch("bert", vocab_size)

    # b) and *two* prediction heads on top that are suited for our task => Language Model finetuning
    lm_prediction_head = BertLMHead(768, vocab_size)
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
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device,
        grad_acc_steps=8,
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    trainer = Trainer.create_or_load_checkpoint(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
        grad_acc_steps=8,
        checkpoint_root_dir=Path("saved_models/train_from_scratch/checkpoints"),
    )
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    train_from_scratch()
