
from farm.infer import Inferencer
from farm.data_handler.utils import write_squad_predictions



lang_model = "deepset/roberta-base-squad2"

# Load model
model = Inferencer.load(model_name_or_path=lang_model, task_type="question_answering", batch_size=90, gpu=True)
result = model.inference_from_file(file="../data/covid/test.json")

write_squad_predictions(
    predictions=result,
    predictions_filename="../data/covid/test.json",
    out_filename="predictions_covid_test_set_one_shot.json"
)
