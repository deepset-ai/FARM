import pprint

from farm.infer import Inferencer


def inference_with_multiprocessing():
    """
    The Inferencers(Inferencer/QAInferencer) create a multiprocessing Pool during the init, if the num_process argument
    is set greater than 0. This helps speed up pre-processing that happens on the CPU, before the model's forward pass
    on GPU(or CPU).

    Having the pool at the Inferencer level allows re-use across multiple inference requests. However, it needs to be
    closed properly to ensure there are no memory-leaks.

    For production environments, the Inferencer object can be wrapped in a try-finally block like in this example to
    ensure the Pool is closed even in the case of errors.
    """

    try:
        model = Inferencer.load("deepset/roberta-base-squad2", batch_size=40, task_type="question_answering", gpu=True)
        QA_input = [
            {
                "qas": ["Who counted the game among the best ever made?"],
                "context": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
            }]
        result = model.inference_from_dicts(dicts=QA_input)[0]

        pprint.pprint(result)
    finally:
        model.close_multiprocessing_pool()


if __name__ == "__main__":
    inference_with_multiprocessing()
