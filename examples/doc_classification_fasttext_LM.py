# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo, StreamingDataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.modeling.wordembedding_utils import Fasttext_converter

def doc_classifcation():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_doc_classification_fasttext")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    n_epochs = 3
    batch_size = 32
    evaluate_every = 100
    # load fasttext from a local path:
    #fasttext_model = "../saved_models/fasttext-german-uncased"
    # or through s3
    fasttext_model = "fasttext-german-uncased"
    do_lower_case = True
    max_features = 10_000 # maximum number of unique words we will transform
    device, n_gpu = initialize_device_settings(use_cuda=True)


    # 1. To make Fasttext work within FARM and with advanced aggregation strategies, we need a fixed vocabulary and associated Wordembeddings
    ft_converter = Fasttext_converter(
        pretrained_model_name_or_path=fasttext_model,
        do_lower_case=do_lower_case,
        data_path=Path("../data/germeval18"),
        train_filename="train.tsv",
        output_path=Path("../saved_models/fasttext-german-uncased-converted"),
        language="German",
        max_features=max_features)
    # We convert the data to have fixed size vocab and embeddings
    vocab_counts = ft_converter.convert_on_data()

    # 2. Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=ft_converter.output_path, do_lower_case=do_lower_case)

    # 3. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load GermEval 2018 Data automaticaly if it is not available.
    # GermEval 2018 only has train.tsv and test.tsv dataset - no dev.tsv
    label_list = ["OTHER", "OFFENSE"]
    metric = "f1_macro"

    processor = TextClassificationProcessor(
        tokenizer=tokenizer,
        max_seq_len=128,
        data_dir=ft_converter.data_path,
        label_list=label_list,
        train_filename=ft_converter.train_filename,
        dev_split=0,
        test_filename="test.tsv",
        metric=metric,
        label_column_name="coarse_label"
        )

    # 4. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    #    few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size,
        max_processes=1) # multiprocessing with WordembeddingTokenizer is not optimal - so disable it

    # 5. Create an AdaptiveModel
    # a) which consists of the newly created embedding model as a basis.
    language_model = LanguageModel.load(ft_converter.output_path)
    # b) and a prediction head on top that is suited for our task => Text classification
    # Since we do not have a powerful Transformer based Language Model, we need a slightly deeper NN
    # for going the Classification
    prediction_head = TextClassificationHead(
        layer_dims=[300,600,len(label_list)],
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
        num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # 6. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-3,
        device=device,
        n_batches=len(data_silo.get_data_loader("train")),  #len(data_silo.loaders["train"]),streaming: len(data_silo.get_data_loader("train"))
        n_epochs=n_epochs)

    # 7. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    # 8. Let it grow
    trainer.train()


if __name__ == "__main__":
    doc_classifcation()

# fmt: on
