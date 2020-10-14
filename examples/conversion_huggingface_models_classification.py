from farm.modeling.adaptive_model import AdaptiveModel
from farm.conversion.transformers import Converter
from farm.data_handler.processor import Processor

from farm.infer import Inferencer
import pprint
from transformers.pipelines import pipeline
from pathlib import Path

##############################################
###  From Transformers -> FARM
##############################################
def convert_from_transformers():
    transformers_input_name = "deepset/bert-base-german-cased-hatespeech-GermEval18Coarse"
    farm_output_dir = Path("../saved_models/farm-bert-base-german-cased-hatespeech-GermEval18Coarse")

    # # CASE 1: MODEL
    # # Load model from transformers model hub (-> continue training / compare models / ...)
    model = Converter.convert_from_transformers(transformers_input_name, device="cpu")

    # # Alternative way to load from transformers model hub:
    #model = AdaptiveModel.convert_from_transformers(transformers_input_name, device="cpu", task_type="text_classification")
    # # ... continue as in the other examples e.g. to fine-tune this QA model on your own data
    #
    # # CASE 2: INFERENCER
    # # Load Inferencer from transformers, incl. model & tokenizer (-> just get predictions)
    nlp = Inferencer.load(transformers_input_name, task_type="text_classification")
    #
    # # run predictions
    result = nlp.inference_from_dicts(dicts=[{"text": "Was ein scheiß Nazi!"}])
    pprint.pprint(result)
    nlp.close_multiprocessing_pool()

    # save it
    nlp.save(farm_output_dir)

# ##############################################
# ###  From FARM -> Transformers
# ##############################################
def convert_to_transformers():
    farm_input_dir = Path("../saved_models/farm-bert-base-german-cased-hatespeech-GermEval18Coarse")
    transformers_output_dir = "../saved_models/bert-base-german-cased-hatespeech-GermEval18Coarse"
    #
    # # # load from FARM format
    model = AdaptiveModel.load(farm_input_dir, device="cpu")
    processor = Processor.load_from_dir(farm_input_dir)
    model.connect_heads_with_processor(processor.tasks)

    # convert to transformers
    transformer_model = Converter.convert_to_transformers(model)[0]
    # # Alternative way to convert to transformers:
    #transformer_model = model.convert_to_transformers()[0]

    # save it (note: transformers use str instead of Path objects)
    Path(transformers_output_dir).mkdir(parents=True, exist_ok=True)
    transformer_model.save_pretrained(transformers_output_dir)
    processor.tokenizer.save_pretrained(transformers_output_dir)

    # run predictions (using transformers)
    nlp = pipeline('sentiment-analysis', model=str(transformers_output_dir), tokenizer=str(transformers_output_dir))
    res = nlp("Was ein scheiß Nazi!")
    pprint.pprint(res)

    # # To upload to transformer's model hub run this in bash:
    # # transformers-cli upload  ../saved_models/bert-large-uncased-whole-word-masking-squad2

if __name__ == "__main__":
    convert_from_transformers()
    convert_to_transformers()