from farm.infer import Inferencer


def test_ner_inference():
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]

    model = Inferencer("../save/bert-german-CONLL2003")
    result = model.run_inference(dicts=basic_texts)
    assert result[0]["predictions"][0]["label"] == "ORG"
    assert result[0]["predictions"][0]["start"] == 15
    assert result[0]["predictions"][0]["end"] == 18


def test_qa_inference():

    QA_input = [
            {
                "questions": ["Who counted the game among the best ever made?"],
                "text":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
            },
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }
    ]


    model = Inferencer("../save/bert-base-english-squad2")
    result = model.run_inference(dicts=QA_input)
    assert result[0]["predictions"][0]["label"] == "GameTrailers"


def test_lm_inference():
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
        {"text": "Schartau2 sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin2 Müller spielt Handball in Berlin"},
        {"text": "Schartau3 sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin3 Müller spielt Handball in Berlin"},
    ]

    model = Inferencer("../save/bert-german-CONLL2003")
    result = model.extract_vectors(dicts=basic_texts, extraction_strategy="per_token")
    print(result)
    assert len(result) == 6
    assert result[0]["context"][0] == "Schar"
    assert result[0]["vec"].shape == (128, 768)