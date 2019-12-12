# fmt: off
import logging
import pprint
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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
# ml_logger = MLFlowLogger(tracking_uri="")
ml_logger.init_experiment(experiment_name="german_qa", run_name="ahmeds_model")

#########################
######## Settings
########################
set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
batch_size = 32
max_seq_len = 512
n_epochs = 2
evaluate_every = 5000
# base_LM_model = "bert-base-multilingual-uncased"
base_LM_model = "../saved_models/multi_squad_en_it_de_farm"
train_filename = "squad20/train-v2.0.json"
dev_filename = "squad20/dev-v2.0.json"
# dev_filename = None
save_dir = "../saved_models/multi_squad_en_it_de_farm"
inference_file = "../data/MLQA_V1/test/test-context-en-question-ar.json"
predictions_file = save_dir + "/predictions.json"
full_predictions_file = save_dir + "/full_predictions.json"
inference_multiprocessing = True
train = False
inference = True

if train:
    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=base_LM_model,
                               tokenizer_class="bert")
    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        label_list=label_list,
        metric=metric,
        train_filename=train_filename,
        dev_filename=dev_filename,
        data_dir="../data/",
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(base_LM_model)
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
    model = Inferencer.load(save_dir, batch_size=40, gpu=True)
    full_result = model.inference_from_file(file=inference_file,
                                            use_multiprocessing=inference_multiprocessing)

    for x in full_result:
        print(x)
        print()

    result = {r["id"]: r["preds"][0][0] for r in full_result}
    full_result =  {r["id"]: r["preds"] for r in full_result}

    json.dump(result,
              open(predictions_file, "w"),
              indent=4,
              ensure_ascii=False)
    json.dump(full_result,
              open(full_predictions_file, "w"),
              indent=4,
              ensure_ascii=False)

