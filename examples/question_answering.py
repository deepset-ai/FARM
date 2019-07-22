# fmt: off
import logging

import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.experiment import initialize_optimizer
from farm.utils import set_all_seeds, MLFlowLogger
from farm.infer import Inferencer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ml_logger = MLFlowLogger(tracking_uri="http://80.158.39.167:5000/")
ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")

##########################
########## Settings
##########################
set_all_seeds(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 24*4
n_epochs = 3
evaluate_every = 200
n_gpu = 4
base_LM_model = "bert-base-cased"
train_filename="train-v2.0.json"
dev_filename="dev-v2.0.json"
save_dir = "../save/qa_model_full"

# 1.Create a tokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=base_LM_model, do_lower_case=False
)
# 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
processor = SquadProcessor(
    tokenizer=tokenizer,
    max_seq_len=256,
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
prediction_head = QuestionAnsweringHead(layer_dims=[768, len(processor.label_list)])

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
    n_examples=data_silo.n_samples("train"),
    batch_size=batch_size,
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
# 7. Let it grow! Watch the tracked metrics live on the public mlflow server: http://80.158.39.167:5000/
model = trainer.train(model)

# 8. Hooray! You have a model. Store it:
save_dir = "save/bert-english-qa-tutorial"
model.save(save_dir)
processor.save(save_dir)

# 9. Load it & harvest your fruits (Inference)
QA_input = [
    {
        "paragraphs": [
            {
                "qas": [
                    {
                        "question": "When did Beyonce start becoming popular?",
                        "id": "123",
                    }
                ],
                "context": 'Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            }
        ]
    }
]
model = Inferencer(save_dir)
result = model.run_inference(dicts=QA_input)
print(result)
