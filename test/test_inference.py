import pytest
import numpy as np
from farm.infer import Inferencer


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("multiprocessing_chunksize", [None, 2])
@pytest.mark.parametrize("num_processes", [2, 0, None], scope="session")
def test_qa_format_and_results(adaptive_model_qa, streaming, multiprocessing_chunksize):
    qa_inputs_dicts = [
        {
            "questions": ["In what country is Normandy"],
            "text": "The Normans are an ethnic group that arose in Normandy, a northern region "
            "of France, from contact between Viking settlers and indigenous Franks and Gallo-Romans",
        },
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received "
            "perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic "
            "Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings "
            "and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores "
            "of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the "
            "greatest games ever created.",
        },
    ]
    ground_truths = ["France", "GameTrailers"]

    results = adaptive_model_qa.inference_from_dicts(
        dicts=qa_inputs_dicts,
        multiprocessing_chunksize=multiprocessing_chunksize,
        streaming=streaming,
    )
    # sample results
    # [
    #     {
    #         "task": "qa",
    #         "predictions": [
    #             {
    #                 "question": "In what country is Normandy",
    #                 "question_id": "None",
    #                 "ground_truth": None,
    #                 "answers": [
    #                     {
    #                         "score": 1.1272038221359253,
    #                         "probability": -1,
    #                         "answer": "France",
    #                         "offset_answer_start": 54,
    #                         "offset_answer_end": 60,
    #                         "context": "The Normans gave their name to Normandy, a region in France.",
    #                         "offset_context_start": 0,
    #                         "offset_context_end": 60,
    #                         "document_id": None,
    #                     }
    #                 ]
    #             }
    #         ],
    #     }
    # ]
    predictions = list(results)[0]["predictions"]

    for prediction, ground_truth, qa_input_dict in zip(
        predictions, ground_truths, qa_inputs_dicts
    ):
        assert prediction["question"] == qa_input_dict["questions"][0]
        answer = prediction["answers"][0]
        assert answer["answer"] in answer["context"]
        assert answer["answer"] == ground_truth
        assert (
                {"answer", "score", "probability", "offset_answer_start", "offset_answer_end", "context",
                 "offset_context_start", "offset_context_end", "document_id"}
                == answer.keys()
        )


@pytest.mark.parametrize("num_processes", [0], scope="session")
def test_embeddings_extraction(num_processes):
    # Input
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
        {"text": "Martin Müller spielt Fussball"},
    ]

    # Load model, tokenizer and processor directly into Inferencer
    model = Inferencer.load(
        model_name_or_path="bert-base-german-cased",
        task_type="embeddings",
        gpu=False,
        batch_size=5,
        extraction_strategy="reduce_mean",
        extraction_layer=-2,
        num_processes=num_processes)

    # Get embeddings for input text (you can vary the strategy and layer)
    result = model.inference_from_dicts(dicts=basic_texts)
    assert result[0]["context"] == ['Schar', '##tau', 'sagte', 'dem', 'Tages', '##spiegel', ',', 'dass', 'Fischer', 'ein', 'Id', '##iot', 'ist']
    assert np.isclose(result[0]["vec"][0], 1.50174605e-02)

if __name__ == "__main__":
    test_embeddings_extraction()
