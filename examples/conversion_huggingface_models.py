from farm.modeling.adaptive_model import AdaptiveModel
from farm.infer import Inferencer
import pprint

# Case 1: Convert a model from transformers model hub to an adaptive model
# (-> e.g. to continue training)
#model = AdaptiveModel.convert_from_transformers("ktrapeznikov/albert-xlarge-v2-squad-v2", device="cpu", task_type="question-answering")

# ... continue as in the other examples to e.g. fine-tune this QA model on your own dataset

# Case 2: You can also directly load everything you need into the Inferencer (incl. tokenizer & processor)
# (-> to just use a pretrained model for )

#a = AutoModel.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

#am = AdaptiveModel.convert_from_transformers("ktrapeznikov/albert-xlarge-v2-squad-v2", device="cpu", task_type="qa")

# i = Inferencer.load("dbmdz/bert-large-cased-finetuned-conll03-english", task_type="ner")
i = Inferencer.load("ktrapeznikov/albert-xlarge-v2-squad-v2", task_type="question-answering")
QA_input_api_format = [
    {
        "questions": ["Who counted the game among the best ever made?"],
        "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
    }]
result = i.inference_from_dicts(dicts=QA_input_api_format, rest_api_schema=True)
for x in result:
    pprint.pprint(x)