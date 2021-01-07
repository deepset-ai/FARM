# fmt: off
import logging
import pprint
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NaturalQuestionsProcessor
from farm.file_utils import fetch_archive_from_http
from farm.infer import QAInferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead, TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings


def question_answering():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_natural_questions")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = 24
    n_epochs = 1
    evaluate_every = 500
    lang_model = "deepset/roberta-base-squad2" # start with a model that can already extract answers
    do_lower_case = False # roberta is a cased model
    train_filename = "train_medium.jsonl"
    dev_filename = "dev_medium.jsonl"
    keep_is_impossible = 0.15 # downsample negative examples after data conversion
    downsample_context_size = 300 # reduce length of wikipedia articles to relevant part around the answer

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case, use_fast=False,
    )

    # Add HTML tag tokens to the tokenizer vocabulary, so they do not get split apart
    html_tags = [
                "<Th>","</Th>",
                "<Td>","</Td>",
                "<Tr>","</Tr>",
                "<Li>","</Li>",
                "<P>" ,"</P>",
                "<Ul>","</Ul>",
                "<H1>","</H1>",
                "<H2>","</H2>",
                "<H3>","</H3>",
                "<H4>","</H4>",
                "<H5>", "</H5>",
                "<Td_colspan=",
    ]
    tokenizer.add_tokens(html_tags)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = NaturalQuestionsProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        train_filename=train_filename,
        dev_filename=dev_filename,
        keep_no_answer=keep_is_impossible,
        downsample_context_size=downsample_context_size,
        data_dir=Path("../data/natural_questions"),
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=batch_size, caching=True)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model,n_added_tokens=len(html_tags))
    # b) and in case of Natural Questions we need two Prediction Heads
    #    one for extractive Question Answering
    qa_head = QuestionAnsweringHead()
    #    another one for answering yes/no questions or deciding if the given text passage might contain an answer
    classification_head = TextClassificationHead(num_labels=len(processor.answer_type_list)) # answer_type_list = ["is_impossible", "span", "yes", "no"]
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[qa_head, classification_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token", "per_sequence"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        schedule_opts={"name": "LinearWarmup", "warmup_proportion": 0.2},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
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
    save_dir = Path("../saved_models/roberta-base-squad2-nq")
    model.save(save_dir)
    processor.save(save_dir)


    # TODO make inferencing work again with trained and saved model (currently the inferencer loads a fast Robertatokenizer, which is not supported by NQ)
    # # 9. Since training on the whole NQ corpus requires substantial compute resources we trained and uploaded a model on s3
    # fetch_archive_from_http("https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/models/roberta-base-squad2-nq.zip", output_dir="../saved_models/farm")
    # QA_input = [
    #     {
    #         "qas": ["Did GameTrailers rated Twilight Princess as one of the best games ever created?"],
    #         "context":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
    #     }
    # ]
    #
    # model = QAInferencer.load(model_name_or_path="../saved_models/farm/roberta-base-squad2-nq", batch_size=batch_size, gpu=True)
    # result = model.inference_from_dicts(dicts=QA_input, return_json=False) # result is a list of QAPred objects
    #
    # print(f"\nQuestion: Did GameTrailers rated Twilight Princess as one of the best games ever created?"
    #       f"\nAnswer from model: {result[0].prediction[0].answer}")
    # model.close_multiprocessing_pool()

if __name__ == "__main__":
    question_answering()


