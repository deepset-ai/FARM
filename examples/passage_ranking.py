# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import RegressionProcessor, TextPairClassificationProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import RegressionHead, TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings, reformat_msmarco_train, reformat_msmarco_dev, write_msmarco_results
from farm.evaluation.msmarco_passage_farm import msmarco_evaluation


def text_pair_classification():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_text_pair_classification")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 2
    batch_size = 64
    evaluate_every = 500
    lang_model = "bert-base-cased"
    label_list = ["0", "1"]
    train_filename = "train.tsv"
    dev_filename = "dev_200k.tsv"

    # The source data can be found here https://github.com/microsoft/MSMARCO-Passage-Ranking
    generate_data = False
    data_dir = Path("../data/msmarco_passage")
    predictions_raw_filename = "predictions_raw.txt"
    predictions_filename = "predictions.txt"
    train_source_filename = "triples.train.1m.tsv"
    qrels_filename = "qrels.dev.tsv"
    queries_filename = "queries.dev.tsv"
    passages_filename = "collection.tsv"
    top1000_filename = "top1000.dev"

    # 0. Preprocess and save MSMarco data in a format that can be ingested by FARM models. Only needs to be done once!
    # The final format is a tsv file with 3 columns (text, text_b and label)
    if generate_data:
        reformat_msmarco_train(data_dir / train_source_filename,
                               data_dir / train_filename)
        reformat_msmarco_dev(data_dir / queries_filename,
                             data_dir / passages_filename,
                             data_dir / qrels_filename,
                             data_dir / top1000_filename,
                             data_dir / dev_filename)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    #    Evaluation during training will be performed on a slice of the train set
    #    We will be using the msmarco dev set as our final evaluation set
    processor = TextPairClassificationProcessor(tokenizer=tokenizer,
                                                label_list=label_list,
                                                train_filename=train_filename,
                                                test_filename=None,
                                                dev_split=0.001,
                                                max_seq_len=128,
                                                data_dir=data_dir,
                                                delimiter="\t")

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task
    prediction_head = TextClassificationHead(num_labels=len(label_list),
                                             class_weights=data_silo.calculate_class_weights(
                                                 task_name="text_classification"),
                                             )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=1e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("saved_models/passage_ranking_model")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    #    Add your own text adapted to the dataset you provide
    model = Inferencer.load(save_dir, gpu=True, max_seq_len=128, batch_size=128)
    result = model.inference_from_file(data_dir / dev_filename)

    write_msmarco_results(result, save_dir / predictions_raw_filename)

    msmarco_evaluation(preds_file=save_dir / predictions_raw_filename,
                       dev_file=data_dir / dev_filename,
                       qrels_file=data_dir / qrels_filename,
                       output_file=save_dir / predictions_filename)

    model.close_multiprocessing_pool()


if __name__ == "__main__":
    text_pair_classification()

# fmt: on
