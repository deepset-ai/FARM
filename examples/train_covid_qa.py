import logging
import json
from pathlib import Path

from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer, EarlyStopping
from farm.infer import Inferencer
from farm.data_handler.utils import write_squad_predictions
from farm.evaluation.metrics import squad

def question_answering_covid_qa():
    ##########################
    ########## Logging
    ##########################
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # reduce verbosity from transformers library
    logging.getLogger('transformers').setLevel(logging.WARNING)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    # for local logging instead:
    # ml_logger = MLFlowLogger(tracking_uri="logs")
    ml_logger.init_experiment(experiment_name="Covid_QA", run_name="2")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=41)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 3
    batch_size = 24
    evaluate_every = 100
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False
    use_amp = None

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        train_filename="train.json",
        dev_filename=None,
        dev_split=0.1,
        test_filename="test.json",
        data_dir=Path("../data/covid"),
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = QuestionAnsweringHead()

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        use_amp=use_amp)

    save_dir = save_dir = Path("saved_models/roberta-qa-covid")
    # An early stopping instance can be used to save the model that performs best on the dev set
    # according to some metric and stop training when no improvement is happening for some iterations.
    earlystopping = EarlyStopping(
        metric="f1", mode="max",  # use the metric from our own metrics function instead of loss
        save_dir=save_dir,  # where to save the best model
        patience=4  # number of evaluations to wait for improvement before terminating the training
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
        early_stopping=earlystopping, )

    # train it
    trainer.train()

    save_dir = Path("../saved_models/roberta-qa-covid")
    model.save(save_dir)
    processor.save(save_dir)

    model = Inferencer.load(save_dir, batch_size=40, gpu=True)

    result = model.inference_from_file(file="../data/covid/test.json")

    write_squad_predictions(
        predictions=result,
        predictions_filename="../data/covid/test.json",
        out_filename="predictions_covid_test_set_fine-tuned.json"
    )
if __name__ == "__main__":
    question_answering_covid_qa()