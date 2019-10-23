# fmt: off
import logging
import pprint

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")

##########################
########## Settings
##########################
set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
batch_size = 24
n_epochs = 2
evaluate_every = 500
base_LM_model = "bert-base-cased"
train_filename="train-v2.0.json"
dev_filename="dev-v2.0.json"

# 1.Create a tokenizer
tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=base_LM_model, do_lower_case=False
)
# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
label_list = ["start_token", "end_token"]
metric = "squad"
processor = SquadProcessor(
    tokenizer=tokenizer,
    max_seq_len=256,
    label_list=label_list,
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
    learning_rate=1e-5,
    warmup_proportion=0.2,
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
save_dir = "../saved_models/bert-english-qa-tutorial"
model.save(save_dir)
processor.save(save_dir)

# 9. Load it & harvest your fruits (Inference)
QA_input = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]

model = Inferencer.load(save_dir)
result = model.inference_from_dicts(dicts=QA_input)

for x in result:
    pprint.pprint(x)
