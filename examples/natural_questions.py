# fmt: off
import logging
import os
import pprint
from pathlib import Path

from farm.data_handler.data_silo import StreamingDataSilo
from farm.data_handler.processor import NaturalQuestionsProcessor
from farm.data_handler.utils import write_squad_predictions
from farm.infer import Inferencer
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
        level=logging.INFO,
    )

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_question_answering")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    batch_size = 20
    n_epochs = 1
    evaluate_every = 100
    #lang_model = "roberta-base"
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False # roberta is a cased model
    train_filename = "train_medium.jsonl"
    dev_filename = "dev_medium.jsonl"
    keep_is_impossible = 0.15 # downsample negative examples
    downsample_context_size = 300 # downsample negative examples before processing

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
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
                "<H4>","</H4>",]
    tokenizer.add_tokens(html_tags)


    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = NaturalQuestionsProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        train_filename=train_filename,
        dev_filename=dev_filename,
        keep_is_impossible=keep_is_impossible,
        downsample_context_size=downsample_context_size,
        data_dir=Path("../data/natural_questions"),
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    # NOTE: In FARM, the dev set metrics differ from test set metrics in that they are calculated on a token level instead of a word level
    streaming_data_silo = StreamingDataSilo(processor=processor, batch_size=batch_size, dataloader_workers=8)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model,n_added_tokens=len(html_tags))
    # b) and a prediction head on top that is suited for our task => Question Answering
    qa_head = QuestionAnsweringHead()
    classification_head = TextClassificationHead(num_labels=len(processor.answer_type_list))

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
        n_batches=len(streaming_data_silo.get_data_loader("train")),
        n_epochs=n_epochs,
        device=device
    )
    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=streaming_data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )
    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("../saved_models/bert_nq")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    QA_input = [
        {
            "qas": ["Who counted the game among the best ever made?"],
            "context":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }
    ]

    model = Inferencer.load(save_dir, batch_size=batch_size, gpu=True)
    result = model.inference_from_dicts(dicts=QA_input)

    pprint.pprint(result)

if __name__ == "__main__":
    question_answering()
