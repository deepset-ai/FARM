# fmt: off
import logging
import pprint
import json

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.eval import Evaluator
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")


#########################
######## Settings
########################
set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
batch_size = 18
n_epochs = 2
evaluate_every = 5
base_LM_model = "bert-base-uncased"
train_filename="subsets/train_medium-v2.0.json"
dev_filename="subsets/dev_medium-v2.0.json"
save_dir = "../saved_models/qa_model_medium"
inference_file = "../data/squad20/subsets/dev_medium-v2.0.json"
predictions_file = save_dir + "/predictions.json"
full_predictions_file = save_dir + "/full_predictions.json"
train = False
inference = True

if train:
    # 1.Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_LM_model,
        do_lower_case=True
    )
    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        labels=label_list,
        metric=metric,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=None,
        data_dir="../data/squad20",
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = Bert.load(base_LM_model)
    # b) and a prediction head on top that is suited for our task => Question Answering
    prediction_head = QuestionAnsweringHead(layer_dims=[768, len(label_list)])

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    # 5. Create an optimizer
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        warmup_proportion=0.1,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
    )
    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device,
    )
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    model = trainer.train(model)

    # 8. Hooray! You have a model. Store it:
    model.save(save_dir)
    processor.save(save_dir)


if inference:
    model = Inferencer.load(save_dir, batch_size=32, gpu=True)
    full_result = model.inference_from_file(file=inference_file)

    for x, y in full_result.items():
        pprint.pprint(x)
        pprint.pprint(y)

    result = {k: v[0][0] for k, v in full_result.items()}

    json.dump(result,
              open(predictions_file, "w"),
              indent=4,
              ensure_ascii=False)
    json.dump(full_result,
              open(full_predictions_file, "w"),
              indent=4,
              ensure_ascii=False)

