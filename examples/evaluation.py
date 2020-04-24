from farm.utils import initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor, SquadProcessor
from farm.data_handler.data_silo import DataSilo
from farm.eval import Evaluator
from farm.modeling.adaptive_model import AdaptiveModel
from pathlib import Path

def evaluate_classification():
    device, n_gpu = initialize_device_settings(use_cuda=True)
    lang_model = "deepset/bert-base-german-cased-sentiment-Germeval17"
    do_lower_case = False
    batch_size = 100

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["negative", "neutral", "positive"]
    metric = "f1_macro"
    processor = TextClassificationProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        train_filename=None,
        dev_filename=None,
        dev_split=0,
        test_filename="test_TIMESTAMP1.tsv",
        data_dir=Path("../data/germeval17"),
    )

    # 3. Create a DataSilo that loads dataset, provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an Evaluator
    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )

    # 5. Load model
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="text_classification")
    # use "load" if you want to use a local model that was trained with FARM
    # model = AdaptiveModel.load(lang_model, device=device)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    # 6. Run the Evaluator
    results = evaluator.eval(model)
    f1_score = results[0]["f1_macro"]
    print("Macro-averaged F1-Score:", f1_score)


def evaluate_question_answering():
    device, n_gpu = initialize_device_settings(use_cuda=True)
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = False
    batch_size = 100

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list=label_list,
        metric=metric,
        train_filename=None,
        dev_filename=None,
        dev_split=0,
        test_filename="dev-v2.0.json",
        data_dir=Path("../data/squad20"),
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads dataset, provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an Evaluator
    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )

    # 5. Load model
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="question_answering")
    # use "load" if you want to use a local model that was trained with FARM
    # model = AdaptiveModel.load(lang_model, device=device)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    # 6. Run the Evaluator
    results = evaluator.eval(model)
    f1_score = results[0]["f1"]
    em_score = results[0]["EM"]
    print("F1-Score:", f1_score)
    print("Exact Match Score:", em_score)


if __name__ == "__main__":
    #evaluate_classification()
    evaluate_question_answering()