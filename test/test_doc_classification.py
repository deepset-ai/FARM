import logging
from pprint import pprint

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

def test_doc_classification(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 8
    evaluate_every = 5
    lang_model = "bert-base-german-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir="samples/doc_class",
                                            train_filename="train-sample.tsv",
                                            label_list=["OTHER", "OFFENSE"],
                                            metric="f1_macro",
                                            dev_filename=None,
                                            test_filename=None,
                                            dev_split=0.1,
                                            label_column_name="coarse_label")

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    language_model = Bert.load(lang_model)
    prediction_head = TextClassificationHead(layer_dims=[768, len(processor.tasks["text_classification"]["label_list"])])
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1)

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device)

    model = trainer.train(model)

    save_dir = "testsave/doc_class"
    model.save(save_dir)
    processor.save(save_dir)

    basic_texts = [
        {"text": "Martin Müller spielt Handball in Berlin."},
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei."},
        {"text": "Franzosen verteidigen 2:1-Führung – Kritische Stimmen zu Schwedens Superstar"},
        {"text": "Neues Video von Designern macht im Netz die Runde"},
        {"text": "23-jähriger Brasilianer muss vier Spiele pausieren – Entscheidung kann noch angefochten werden"},
        {"text": "Aufständische verwendeten Chemikalie bei Gefechten im August."},
        {"text": "Bewährungs- und Geldstrafe für 26-Jährigen wegen ausländerfeindlicher Äußerung"},
        {"text": "ÖFB-Teamspieler nur sechs Minuten nach seinem Tor beim 1:1 gegen Sunderland verletzt ausgewechselt"},
        {"text": "Ein 31-jähriger Polizist soll einer 42-Jährigen den Knöchel gebrochen haben"},
        {"text": "18 Menschen verschleppt. Kabul – Nach einem Hubschrauber-Absturz im Norden Afghanistans haben Sicherheitskräfte am Mittwoch versucht"}
    ]
    #TODO enable loading here again after we have finished migration towards "processor.tasks"
    #inf = Inferencer.load(save_dir)
    inf = Inferencer(model=model, processor=processor)
    result = inf.run_inference(dicts=basic_texts)
    assert result[0]["predictions"][0]["label"] == "OTHER"
    assert abs(result[0]["predictions"][0]["probability"] - 0.7) <= 0.1

    loaded_processor = TextClassificationProcessor.load_from_dir(save_dir)
    inf2 = Inferencer(model=model, processor=loaded_processor)
    result_2 = inf2.run_inference(dicts=basic_texts)
    pprint(list(zip(result, result_2)))
    for r1, r2 in list(zip(result, result_2)):
        assert r1 == r2

# if(__name__=="__main__"):
#     test_doc_classification()