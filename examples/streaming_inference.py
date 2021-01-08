from farm.infer import Inferencer


def streaming_inference_example():
    """
    The FARM Inferencer has a high performance non-blocking streaming mode for large scale inference use cases. With
    this mode, the dicts parameter can optionally be a Python generator object that yield dicts, thus avoiding loading
    dicts in memory. The inference_from_dicts() method returns a generator that yield predictions. To use streaming,
    set the streaming param to True and determine optimal multiprocessing_chunksize by performing speed benchmarks.
    """

    model_name_or_path = "deepset/bert-base-cased-squad2"
    inferencer = Inferencer.load(model_name_or_path=model_name_or_path, task_type="question_answering", num_processes=8)

    dicts = sample_dicts_generator()  # it can be a list of dicts or a generator object
    results = inferencer.inference_from_dicts(dicts, streaming=True, multiprocessing_chunksize=20)

    for prediction in results:  # results is a generator object that yields predictions
        print(prediction)

    inferencer.close_multiprocessing_pool()


def sample_dicts_generator():
    """
    This is a sample dicts generator. Some exemplary use cases:

    * read chunks of text from large files iteratively and generate inference predictions
    * connect with external datasources, eg, a Elasticsearch Scroll API that reads all documents from a given index
    * building a streaming microservice that reads from Kafka

    :return: a generator that yield dicts
    :rtype: iter
    """
    qa_input = {
        "questions": ["Who counted the game among the best ever made?"],
        "text": "Twilight Princess was released to universal critical acclaim and commercial success. "
                   "It received perfect scores from major publications such as 1UP.com, Computer and Video Games, "
                   "Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators "
                   "GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii "
                   "version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called "
                   "it one of the greatest games ever created.",
    }

    for i in range(100000):
        yield qa_input


if __name__ == "__main__":
    streaming_inference_example()
