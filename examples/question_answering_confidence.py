import logging
import torch
from pathlib import Path

from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.adaptive_model import AdaptiveModel
from farm.infer import QAInferencer
from farm.eval import Evaluator
from farm.evaluation.metrics import metrics_per_bin


def question_answering_confidence():
    ##########################
    ########## Logging
    ##########################
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # reduce verbosity from transformers library
    logging.getLogger('transformers').setLevel(logging.WARNING)

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)

    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False
    batch_size = 80

    data_dir = Path("../data/squad20")
    # We use the same file for dev and test set only for demo purposes
    dev_filename = "dev-v2.0.json"
    test_filename = "dev-v2.0.json"
    accuracy_at = 3 # accuracy at n is useful for answers inside long documents


    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=["start_token", "end_token"],
        metric="squad",
        train_filename=None,
        dev_filename=dev_filename,
        test_filename=test_filename,
        data_dir=data_dir,
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)


    # 4. Load pre-trained question-answering model
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="question_answering")
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)
    # Number of predictions the model will make per Question.
    # The multiple predictions are used for evaluating top n recall.
    model.prediction_heads[0].n_best = accuracy_at

    # 5. The calibration of model confidence scores sets one parameter, which is called temperature and can be accessed through the prediction_head.
    # This temperature is applied to each logit in the forward pass, where each logit is divided by the temperature.
    # A softmax function is applied to the logits afterward to get confidence scores in the range [0,1].
    # A temperature larger than 1 decreases the model’s confidence scores.
    logger.info(f"Parameter used for temperature scaling of model confidence scores: {model.prediction_heads[0].temperature_for_confidence}")

    # 6a. We can either manually set the temperature (default value is 1.0)...
    model.prediction_heads[0].temperature_for_confidence = torch.nn.Parameter((torch.ones(1) * 1.0).to(device=device))

    # 6b. ...or we can run the evaluator on the dev set and use it to calibrate confidence scores with a technique called temperature scaling.
    # It will align the confidence scores with the model's accuracy based on the dev set data by tuning the temperature parameter.
    # During the calibration, this parameter is automatically set internally as an attribute of the prediction head.
    evaluator_dev = Evaluator(
        data_loader=data_silo.get_data_loader("dev"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    result_dev = evaluator_dev.eval(model, return_preds_and_labels=True, calibrate_conf_scores=True)
    # evaluator_dev.log_results(result_dev, "Dev", logging=False, steps=len(data_silo.get_data_loader("dev")))

    # 7. Optionally, run the evaluator on the test set to see how well the confidence scores are aligned with the model's accuracy
    evaluator_test = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    result_test = evaluator_test.eval(model, return_preds_and_labels=True)[0]
    logger.info("Grouping predictions by confidence score and calculating metrics for each bin.")
    em_per_bin, confidence_per_bin, count_per_bin = metrics_per_bin(result_test["preds"], result_test["labels"], num_bins=10)
    for bin_number in range(10):
        logger.info(f"Bin {bin_number} - exact match: {em_per_bin[bin_number]}, average confidence score: {confidence_per_bin[bin_number]}")

    # 8. Hooray! You have a model with calibrated confidence scores.
    # Store the model and the temperature parameter will be stored automatically as an attribute of the prediction head.
    save_dir = Path("../saved_models/qa-confidence-tutorial")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. When making a prediction with the calibrated model, we could filter out predictions where the model is not confident enough
    # To this end, load the stored model, which will automatically load the stored temperature parameter.
    # The confidence scores are automatically adjusted based on this temperature parameter.
    # For each prediction, we can check the model's confidence and decide whether to output the prediction or not.
    inferencer = QAInferencer.load(save_dir, batch_size=40, gpu=True)
    logger.info(f"Loaded model with stored temperature: {inferencer.model.prediction_heads[0].temperature_for_confidence}")

    QA_input = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]
    result = inferencer.inference_from_dicts(dicts=QA_input, return_json=False)[0]
    if result.prediction[0].confidence > 0.9:
        print(result.prediction[0].answer)
    else:
        print("The confidence is not high enough to give an answer.")


if __name__ == "__main__":
    question_answering_confidence()
