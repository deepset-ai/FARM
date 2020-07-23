# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

def doc_classifcation():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_doc_classification")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    n_epochs = 1
    batch_size = 32
    evaluate_every = 100
    lang_model = "bert-base-german-cased"
    do_lower_case = False
    # or a local path:
    # lang_model = Path("../saved_models/farm-bert-base-cased")
    use_amp = None

    #############################################
    # CUSTOM OPTIMIZER & LR SCHEDULE
    #############################################
    # learning rate schedules from transformers
    schedule_opts = {"name": "LinearWarmup", "warmup_proportion": 0.4}
    # schedule_opts = {"name": "Constant"}
    # schedule_opts = {"name": "CosineWarmup", "warmup_proportion": 0.4}
    # schedule_opts = {"name": "CosineWarmupWithRestarts", "warmup_proportion": 0.4}

    # or from native pytorch (see https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html for all options)
    # schedule_opts = {"name": "StepLR", "step_size": 30, "gamma": 0.1}
    # schedule_opts = {"name": "ReduceLROnPlateau", "mode": 'min', "factor": 0.1, "patience":10}

    # optimizers from pytorch (see https://pytorch.org/docs/stable/optim.html for all options)
    optimizer_opts = {"name": "SGD", "momentum": 0.0}

    # or from apex (see https://github.com/NVIDIA/apex/tree/master/apex/optimizers for all options)
    # optimizer_opts = {"name": "FusedLAMB", "bias_correction": True}

    # or from transformers (default in FARM)
    #optimizer_opts = {"name": "TransformersAdamW", "correct_bias": False, "weight_decay": 0.01}
    #############################################


    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=use_amp)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load GermEval 2018 Data automaticaly if it is not available.
    # GermEval 2018 only has train.tsv and test.tsv dataset - no dev.tsv

    label_list = ["OTHER", "OFFENSE"]
    metric = "f1_macro"

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir=Path("../data/germeval18"),
                                            label_list=label_list,
                                            metric=metric,
                                            label_column_name="coarse_label"
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    #    few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
        num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=5e-3,
        optimizer_opts=optimizer_opts,
        schedule_opts=schedule_opts,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        use_amp=use_amp)

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
    save_dir = Path("saved_models/bert-german-doc-tutorial")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]
    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    print(result)
    model.close_multiprocessing_pool()


if __name__ == "__main__":
    doc_classifcation()

# fmt: on
