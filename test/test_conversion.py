from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.infer import Inferencer
import pprint
from transformers.pipelines import pipeline
from transformers.modeling_auto import AutoModelForQuestionAnswering
import os
from pathlib import Path

import logging

def test_conversion_adaptive_model(caplog):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    model = AdaptiveModel.convert_from_transformers("deepset/bert-base-cased-squad2", device="cpu", task_type="question_answering")
    transformer_model = model.convert_to_transformers()
    transformer_model2 = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    # compare weights
    for p1, p2 in zip(transformer_model.parameters(), transformer_model2.parameters()):
        assert(p1.data.ne(p2.data).sum() == 0)


def test_conversion_inferencer(caplog):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    # input
    question = "Why is model conversion important?"
    text = "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."


    # Load from model hub
    model = "deepset/bert-base-cased-squad2"
    nlp = Inferencer.load(model, task_type="question_answering", num_processes=0)

    assert nlp.processor.tokenizer.basic_tokenizer.do_lower_case == False

    QA_input = [{"questions": [question], "text": text}]
    result_farm = nlp.inference_from_dicts(dicts=QA_input)
    answer_farm = result_farm[0]["predictions"][0]["answers"][0]["answer"]
    assert answer_farm == 'gives freedom to the user'

    # save it
    farm_model_dir = Path("testsave/bert-conversion-test")
    nlp.save(farm_model_dir)

    # load from disk in FARM format
    model = AdaptiveModel.load(farm_model_dir, device="cpu")
    tokenizer = Tokenizer.load(farm_model_dir)

    # convert to transformers
    transformer_model = model.convert_to_transformers()

    # save it (Note: transformers uses strings rather than Path objects)
    model_dir = "testsave/bert-conversion-test-hf"
    os.makedirs(model_dir, exist_ok=True)
    transformer_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # run predictions (using transformers)
    nlp = pipeline('question-answering', model=model_dir, tokenizer=model_dir)
    result_transformers = nlp({
        'question': question,
        'context': text
    })
    answer_transformers = result_transformers["answer"]
    assert answer_farm == answer_transformers

if __name__ == "__main__":
    test_conversion_inferencer(None)
    test_conversion_adaptive_model(None)
