# fmt: off
import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import CONLLProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_minimal_example_ner")

##########################
########## Settings
##########################
set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
n_epochs = 4
batch_size = 32
evaluate_every = 50
lang_model = "bert-base-german-cased"

# 1.Create a tokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=lang_model, do_lower_case=False
)

# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
processor = CONLLProcessor(
    tokenizer=tokenizer, max_seq_len=128, data_dir="../data/conll03-de"
)

# 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(processor=processor, batch_size=batch_size)

# 4. Create an AdaptiveModel
# a) which consists of a pretrained language model as a basis
language_model = Bert.load(lang_model)
# b) and a prediction head on top that is suited for our task => NER
prediction_head = TokenClassificationHead(layer_dims=[768, len(processor.label_list)])

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
    learning_rate=2e-5,
    warmup_proportion=0.1,
    n_examples=data_silo.n_samples("train"),
    batch_size=batch_size,
    n_epochs=n_epochs,
)

# 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=n_gpu,
    warmup_linear=warmup_linear,
    evaluate_every=evaluate_every,
    device=device,
)

# 7. Let it grow
model = trainer.train(model)

# 8. Hooray! You have a model. Store it:
model.save("saved_models/bert-german-ner-tutorial")
processor.save("saved_models/bert-german-ner-tutorial")


# 9. Load it & harvest your fruits (Inference)
basic_texts = [
    {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
    {"text": "Martin Müller spielt Handball in Berlin"},
]
model = Inferencer.load("save/bert-german-ner-tutorial")
result = model.run_inference(dicts=basic_texts)
print(result)
