from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.conversion.transformers import Converter
from farm.infer import Inferencer
import pprint
from transformers.pipelines import pipeline
import os
from pathlib import Path

##############################################
###  From Transformers -> FARM
##############################################
def convert_from_transformers():
    # CASE 1: MODEL
    # Load model from transformers model hub (-> continue training / compare models / ...)
    model = Converter.convert_from_transformers("deepset/bert-large-uncased-whole-word-masking-squad2", device="cpu")
    #Alternative way to load from transformers model hub:
    #model = AdaptiveModel.convert_from_transformers("deepset/bert-large-uncased-whole-word-masking-squad2", device="cpu", task_type="question_answering")
    # ... continue as in the other examples e.g. to fine-tune this QA model on your own data

    # CASE 2: INFERENCER
    # Load Inferencer from transformers, incl. model & tokenizer (-> just get predictions)
    nlp = Inferencer.load("deepset/bert-large-uncased-whole-word-masking-squad2", task_type="question_answering")

    # run predictions
    QA_input = [{"questions": ["Why is model conversion important?"],
                 "text": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."}]
    result = nlp.inference_from_dicts(dicts=QA_input)
    pprint.pprint(result)
    nlp.close_multiprocessing_pool()

    # save it
    farm_model_dir = Path("../saved_models/bert-english-qa-large")
    nlp.save(farm_model_dir)

##############################################
###  From FARM -> Transformers
##############################################
def convert_to_transformers():
    farm_model_dir = Path("../saved_models/bert-english-qa-large")

    # load from FARM format
    model = AdaptiveModel.load(farm_model_dir, device="cpu")
    tokenizer = Tokenizer.load(farm_model_dir)

    # convert to transformers
    transformer_model = Converter.convert_to_transformers(model)[0]
    #Alternative way to convert to transformers:
    #transformer_model = model.convert_to_transformers()[0]

    # save it (Note: transformers uses strings rather than Path objects)
    model_dir = "../saved_models/bert-large-uncased-whole-word-masking-squad2"
    os.makedirs(model_dir, exist_ok=True)
    transformer_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # run predictions (using transformers)
    nlp = pipeline('question-answering', model=model_dir, tokenizer=model_dir)
    res = nlp({
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    })
    pprint.pprint(res)

    # To upload to transformer's model hub run this in bash:
    # transformers-cli upload  ../saved_models/bert-large-uncased-whole-word-masking-squad2


if __name__ == "__main__":
    convert_from_transformers()
    convert_to_transformers()