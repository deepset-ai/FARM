import logging
from pathlib import Path
import numpy as np

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.infer import Inferencer

def test_qa(caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    batch_size = 2
    n_epochs = 1
    evaluate_every = 4
    base_LM_model = "distilbert-base-uncased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=base_LM_model, do_lower_case=True
    )
    label_list = ["start_token", "end_token"]
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=20,
        doc_stride=10,
        max_query_length=6,
        train_filename="train-sample.json",
        dev_filename="dev-sample.json",
        test_filename=None,
        data_dir=Path("samples/qa"),
        label_list=label_list,
        metric="squad"
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)
    language_model = LanguageModel.load(base_LM_model)
    prediction_head = QuestionAnsweringHead()
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device
    )
    trainer.train()
    save_dir = Path("testsave/qa")
    model.save(save_dir)
    processor.save(save_dir)

    inferencer = Inferencer.load(save_dir, batch_size=2, gpu=False)

    QA_input_api_format = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]
    QA_input_squad = [{"qas":["Who counted the game among the best ever made?"],
                 "context": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                }]


    result = inferencer.inference_from_dicts(dicts=QA_input_squad)
    result_api_format = inferencer.inference_from_dicts(dicts=QA_input_api_format, rest_api_schema=True)

    # top answer
    assert result[0]["preds"][0][0] == result_api_format[0]["predictions"][0]["answers"][0]["answer"]
    # top score
    assert result[0]["preds"][0][3] == result_api_format[0]["predictions"][0]["answers"][0]["score"]


def test_qa_onnx_inference():
    QA_input_api_format = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]
    QA_input_squad = [{"qas":["Who counted the game among the best ever made?"],
                 "context": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
                }]

    base_LM_model = "deepset/bert-base-cased-squad2"

    # Pytorch
    inferencer = Inferencer.load(base_LM_model, batch_size=2, gpu=False, task_type="question_answering")
    result = inferencer.inference_from_dicts(dicts=QA_input_squad)[0]
    result_api_format = inferencer.inference_from_dicts(dicts=QA_input_api_format, rest_api_schema=True)[0]

    # ONNX
    onnx_model_export_path = Path("testsave/onnx-export")
    inferencer.model.convert_to_onnx(onnx_model_export_path)
    inferencer = Inferencer.load(model_name_or_path=onnx_model_export_path, task_type="question_answering")

    result_onnx = inferencer.inference_from_dicts(QA_input_squad)[0]
    result_onnx_api_format = inferencer.inference_from_dicts(QA_input_api_format, rest_api_schema=True)[0]

    # Standard squad format
    for pred in range(len(result["preds"])):
        assert result_onnx["preds"][pred][0] == result["preds"][pred][0] # answer string
        assert result_onnx["preds"][pred][1] == result["preds"][pred][1] # offset start
        assert result_onnx["preds"][pred][2] == result["preds"][pred][2] # offset end
        np.testing.assert_almost_equal(result_onnx["preds"][pred][2], result["preds"][pred][2]) # score

    # API format
    for (onnx, regular) in zip(result_onnx_api_format["predictions"][0]["answers"][0].items(), result_api_format["predictions"][0]["answers"][0].items()):
        # keys
        assert onnx[0] == regular[0]
        # values
        if type(onnx[1]) == float:
            np.testing.assert_almost_equal(onnx[1], regular[1], decimal=4)  # score
        else:
            assert onnx[1] == regular[1]


if(__name__=="__main__"):
    test_qa_onnx_inference()
