# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.eval import Evaluator
from farm.utils import set_all_seeds, initialize_device_settings

def question_answering_model_confidence():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = 24
    eval_batch_size = 4*batch_size
    n_epochs = 1
    evaluate_every = 2000
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False # roberta is a cased model
    train_filename = "train-v2.0-xs.json"
    dev_filename = "dev-v2.0-1st-half.json"
    test_filename = "dev-v2.0-2nd-half.json"
    #train_filename = "train-v2.0.json"
    #dev_filename = "dev-v2.0.json"
    #test_filename = "dev-v2.0.json"


    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(lang_model, do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        #train_filename=train_filename,
        train_filename=None,
        dev_filename=dev_filename,
        test_filename=test_filename,
        data_dir=Path("../data/squad20"),
    )


    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    # NOTE: In FARM, the dev set metrics differ from test set metrics in that they are calculated on a token level instead of a word level
    data_silo = DataSilo(processor=processor, batch_size=batch_size, eval_batch_size=eval_batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="question_answering",
                                                    processor=processor)


    # 5. Create an optimizer
    #model, optimizer, lr_schedule = initialize_optimizer(
    #    model=model,
    #    learning_rate=3e-5,
    #    schedule_opts={"name": "LinearWarmup", "warmup_proportion": 0.2},
    #    n_batches=len(data_silo.loaders["train"]),
    #    n_epochs=n_epochs,
    #    device=device
    #)

    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    #trainer = Trainer(
    #    model=model,
    #    optimizer=optimizer,
    #    data_silo=data_silo,
    #    epochs=n_epochs,
    #    n_gpu=n_gpu,
    #    lr_schedule=lr_schedule,
    #    evaluate_every=evaluate_every,
    #    device=device,
    #)
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    #trainer.train()

    # 8. Hooray! You have a model. Store it:
    #save_dir = Path("../saved_models/bert-english-qa-tutorial")
    #model.save(save_dir)
    #processor.save(save_dir)

    evaluator_dev = Evaluator(
        data_loader=data_silo.get_data_loader("dev"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    evaluator_test = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    result = evaluator_dev.eval(model, return_preds_and_labels=True, update_temp=True)
    evaluator_test.log_results(result, "Dev", logging=False, steps=len(data_silo.get_data_loader("dev")))
    result = evaluator_test.eval(model, return_preds_and_labels=True, update_temp=False)
    evaluator_test.log_results(result, "Test", logging=False, steps=len(data_silo.get_data_loader("test")))

if __name__ == "__main__":
    question_answering_model_confidence()
