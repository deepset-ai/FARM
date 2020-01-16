from farm.modeling.adaptive_model import AdaptiveModel
from transformers.modeling_auto import AutoModel, AutoModelForQuestionAnswering
from farm.infer import Inferencer

#a = AutoModel.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

#am = AdaptiveModel.convert_from_transformers("ktrapeznikov/albert-xlarge-v2-squad-v2", device="cpu", task_type="qa")

i = Inferencer.load("ktrapeznikov/albert-xlarge-v2-squad-v2", task_type="question-answering")
QA_input = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "context":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
        }]
i.inference_from_dicts(dicts=QA_input, max_processes=1)