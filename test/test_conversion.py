from transformers import BertForSequenceClassification, BertForTokenClassification

from farm.modeling.adaptive_model import AdaptiveModel
from farm.conversion.transformers import Converter
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead, TokenClassificationHead, QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.infer import Inferencer
from transformers.pipelines import pipeline
from transformers import AutoModelForQuestionAnswering, AutoModelWithLMHead, \
    AutoModelForSequenceClassification, AutoModelForTokenClassification
import os
from pathlib import Path
import pytest

@pytest.mark.conversion
def test_conversion_adaptive_model_qa():
    farm_model = Converter.convert_from_transformers("deepset/bert-base-cased-squad2", device="cpu")
    transformer_model = farm_model.convert_to_transformers()[0]
    transformer_model2 = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    # compare weights
    for p1, p2 in zip(transformer_model.parameters(), transformer_model2.parameters()):
        assert (p1.data.ne(p2.data).sum() == 0)

@pytest.mark.conversion
def test_conversion_adaptive_model_lm():
    farm_model = Converter.convert_from_transformers("bert-base-german-cased", device="cpu")
    transformer_model = farm_model.convert_to_transformers()[0]
    transformer_model2 = AutoModelWithLMHead.from_pretrained("bert-base-german-cased")
    # compare weights
    for p1, p2 in zip(transformer_model.parameters(), transformer_model2.parameters()):
        assert (p1.data.ne(p2.data).sum() == 0)

@pytest.mark.conversion
def test_conversion_adaptive_model_classification():
    farm_model = Converter.convert_from_transformers("deepset/bert-base-german-cased-hatespeech-GermEval18Coarse", device="cpu")
    transformer_model = farm_model.convert_to_transformers()[0]
    transformer_model2 = AutoModelForSequenceClassification.from_pretrained("deepset/bert-base-german-cased-hatespeech-GermEval18Coarse")
    # compare weights
    for p1, p2 in zip(transformer_model.parameters(), transformer_model2.parameters()):
        assert (p1.data.ne(p2.data).sum() == 0)

@pytest.mark.conversion
def test_conversion_adaptive_model_ner():
    farm_model = Converter.convert_from_transformers("dslim/bert-base-NER", device="cpu")
    transformer_model = farm_model.convert_to_transformers()[0]
    transformer_model2 = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    # compare weights
    for p1, p2 in zip(transformer_model.parameters(), transformer_model2.parameters()):
        assert (p1.data.ne(p2.data).sum() == 0)

@pytest.mark.conversion
def test_conversion_inferencer_qa():
    # input
    question = "Why is model conversion important?"
    text = "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."

    # Load from model hub
    model = "deepset/bert-base-cased-squad2"
    nlp = Inferencer.load(model, task_type="question_answering", num_processes=0)

    assert nlp.processor.tokenizer.do_lower_case == False
    assert nlp.processor.tokenizer.is_fast == True

    QA_input = [{"questions": [question], "text": text}]
    result_farm = nlp.inference_from_dicts(dicts=QA_input)
    answer_farm = result_farm[0]["predictions"][0]["answers"][0]["answer"]
    assert answer_farm == 'gives freedom to the user'

    # save it
    farm_model_dir = Path("testsave/bert-conversion-test")
    nlp.save(farm_model_dir)

    # free RAM
    del nlp

    # load from disk in FARM format
    model = AdaptiveModel.load(farm_model_dir, device="cpu")
    tokenizer = Tokenizer.load(farm_model_dir)

    # convert to transformers
    transformer_model = Converter.convert_to_transformers(model)[0]

    # free RAM
    del model

    # save it (Note: transformers uses strings rather than Path objects)
    model_dir = "testsave/bert-conversion-test-hf"
    os.makedirs(model_dir, exist_ok=True)
    transformer_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    del transformer_model
    del tokenizer

    # run predictions (using transformers)
    nlp = pipeline('question-answering', model=model_dir, tokenizer=model_dir)
    result_transformers = nlp({
        'question': question,
        'context': text
    })
    answer_transformers = result_transformers["answer"]
    assert answer_farm == answer_transformers
    del nlp

@pytest.mark.conversion
def test_conversion_inferencer_classification():
    # input
    text = "Das ist blöd."

    # Load from model hub
    model = "deepset/bert-base-german-cased-hatespeech-GermEval18Coarse"
    nlp = Inferencer.load(model, task_type="text_classification", num_processes=0)

    assert nlp.processor.tokenizer.do_lower_case == False
    assert nlp.processor.tokenizer.is_fast == True

    input = [{"text": text}]
    result_farm = nlp.inference_from_dicts(dicts=input)
    pred_farm = result_farm[0]["predictions"][0]["label"]
    assert pred_farm == 'OFFENSE'

    # save it
    farm_model_dir = Path("testsave/bert-conversion-test-hf")
    nlp.save(farm_model_dir)
    del nlp

    # load from disk in FARM format
    model = AdaptiveModel.load(farm_model_dir, device="cpu")
    tokenizer = Tokenizer.load(farm_model_dir)

    # convert to transformers
    transformer_model = Converter.convert_to_transformers(model)[0]
    del model

    # save it (Note: transformers uses strings rather than Path objects)
    model_dir = "testsave/bert-conversion-test-hf"
    os.makedirs(model_dir, exist_ok=True)
    transformer_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    del transformer_model
    del tokenizer

    # run predictions (using transformers)
    nlp = pipeline('sentiment-analysis', model=model_dir, tokenizer=model_dir)
    result_transformers = nlp(text)
    pred_transformers = result_transformers[0]["label"]
    assert pred_farm == pred_transformers
    del nlp

@pytest.mark.conversion
def test_conversion_inferencer_ner():
    # input
    text = "Paris is a town in France."

    # Load from model hub
    model = "dslim/bert-base-NER"
    nlp = Inferencer.load(model, task_type="ner", num_processes=0)

    assert nlp.processor.tokenizer.do_lower_case == False
    assert nlp.processor.tokenizer.is_fast == True

    input = [{"text": text}]
    result_farm = nlp.inference_from_dicts(dicts=input)
    pred_farm = result_farm[0]["predictions"]
    assert pred_farm[0]["label"] == 'LOC'
    assert pred_farm[1]["label"] == 'LOC'
    assert len(pred_farm) == 2

    # save it
    farm_model_dir = Path("testsave/bert-conversion-test-hf")
    nlp.save(farm_model_dir)
    del nlp

    # load from disk in FARM format
    model = AdaptiveModel.load(farm_model_dir, device="cpu")
    tokenizer = Tokenizer.load(farm_model_dir)

    # convert to transformers
    transformer_model = Converter.convert_to_transformers(model)[0]
    del model

    # save it (Note: transformers uses strings rather than Path objects)
    model_dir = "testsave/bert-conversion-test-hf"
    os.makedirs(model_dir, exist_ok=True)
    transformer_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    del transformer_model
    del tokenizer

    # run predictions (using transformers)
    nlp = pipeline('ner', model=model_dir, tokenizer=model_dir)
    result_transformers = nlp(text)
    assert result_transformers[0]["entity"] == 'B-LOC'
    assert result_transformers[1]["entity"] == 'B-LOC'
    assert len(result_transformers) == 2
    del nlp

@pytest.mark.conversion
def test_multiple_prediction_heads():
    model = "bert-base-german-cased"
    lm = LanguageModel.load(model)
    ph1 = TextClassificationHead(num_labels=3, label_list=["negative", "neutral", "positive"])
    ph2 = TokenClassificationHead(num_labels=3, label_list=["PER", "LOC", "ORG"])
    adaptive_model = AdaptiveModel(language_model=lm, prediction_heads=[ph1, ph2], embeds_dropout_prob=0.1,
                                   lm_output_types="per_token", device="cpu")
    transformer_models = Converter.convert_to_transformers(adaptive_model)
    assert isinstance(transformer_models[0], BertForSequenceClassification)
    assert isinstance(transformer_models[1], BertForTokenClassification)
    del lm
    del transformer_models
    del adaptive_model
