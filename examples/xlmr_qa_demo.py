# fmt: off
import logging
import pprint
from pathlib import Path
import json

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


def xlmr_qa_demo():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="run_xmlr_qa")

    #########################
    ######## Settings
    ########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = 3
    grad_acc_steps = 8
    n_epochs = 2
    evaluate_every = 200
    base_LM_model = "xlm-roberta-large"

    data_dir = Path("../data/squad20")
    train_filename = Path("train-v2.0.json")
    dev_filename = Path("dev-v2.0.json")

    save_dir = Path("../saved_models/xlmr-large-qa")

    inference_file = Path("../data/MLQA_V1/dev/dev-context-de-question-de.json")
    predictions_file = save_dir / "predictions.json"
    full_predictions_file = save_dir / "full_predictions.json"
    max_processes_for_inference = 8
    train = True
    inference = False

    if train:
        # 1.Create a tokenizer
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=base_LM_model)
        # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        label_list = ["start_token", "end_token"]
        metric = "squad"
        processor = SquadProcessor(
            tokenizer=tokenizer,
            max_seq_len=384,
            label_list=label_list,
            metric=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=None,
            data_dir=data_dir,
            dev_split=0.0
        )

        # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
        data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False, max_processes=1)

        # 4. Create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis
        language_model = LanguageModel.load(base_LM_model, n_added_tokens=3)
        # b) and a prediction head on top that is suited for our task => Question Answering
        prediction_head = QuestionAnsweringHead()

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm_output_types=["per_token"],
            device=device,
        )

        # 5. Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=3e-5,
            schedule_opts={"name": "LinearWarmup", "warmup_proportion": 0.2},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            grad_acc_steps=grad_acc_steps,
            device=device
        )

        # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device,
        )
        # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
        trainer.train()

        # 8. Hooray! You have a model. Store it:
        model.save(save_dir)
        processor.save(save_dir)

    if inference:
        model = Inferencer.load(save_dir, batch_size=32, gpu=True)
        full_result = model.inference_from_file(
            file=inference_file,
            max_processes=max_processes_for_inference,
        )

        for x in full_result:
            print(x)
            print()

        result = {r["id"]: r["preds"][0][0] for r in full_result}
        full_result = {r["id"]: r["preds"] for r in full_result}

        json.dump(result,
                  open(predictions_file, "w"),
                  indent=4,
                  ensure_ascii=False)
        json.dump(full_result,
                  open(full_predictions_file, "w"),
                  indent=4,
                  ensure_ascii=False)


if __name__ == "__main__":
    xlmr_qa_demo()
