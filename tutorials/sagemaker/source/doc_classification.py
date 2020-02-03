import argparse
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


def doc_classification(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    ml_logger = MLFlowLogger(tracking_uri="")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_doc_classification")

    set_all_seeds(seed=42)
    save_dir = Path("/opt/ml/model")
    use_amp = None

    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=args.base_lm_model, do_lower_case=False)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load GermEval 2018 Data.
    label_list = ["OTHER", "OFFENSE"]
    metric = "f1_macro"

    processor = TextClassificationProcessor(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        data_dir=Path("../data/germeval18"),
        label_list=label_list,
        metric=metric,
        label_column_name="coarse_label",
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    #    few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=args.batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(args.base_lm_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"), num_labels=len(label_list)
    )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=args.n_epochs,
        use_amp=use_amp,
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=args.evaluate_every,
        device=device,
    )

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size (default: 4)")
    parser.add_argument("--max_seq_len", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument(
        "--base_lm_model",
        type=str,
        default="bert-base-uncased",
        help="base language model to use (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--evaluate_every", type=int, default=100, help="perform evaluation every n steps (default: 100)"
    )
    doc_classification(parser.parse_args())
