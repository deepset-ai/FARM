from farm.modeling.adaptive_model import AdaptiveModel
from farm.infer import Inferencer
import pprint

# CASE 1:
# Convert a model from transformers model hub to an adaptive model
# (-> e.g. to continue training)
model = AdaptiveModel.convert_from_transformers("ktrapeznikov/albert-xlarge-v2-squad-v2", device="cpu", task_type="question-answering")
# ... continue as in the other examples to e.g. fine-tune this QA model on your own dataset

# CASE 2:
# You can also directly load everything you need into the Inferencer (incl. tokenizer & processor)
# (-> just get predictions from a pretrained model)

# Question answering
inf = Inferencer.load("ktrapeznikov/albert-xlarge-v2-squad-v2", task_type="question-answering")
QA_input = [
    {"questions": ["Who counted the game among the best ever made?"],
     "text": "Twilight Princess was released to universal critical acclaim and commercial success. GameTrailers in their review called it one of the greatest games ever created."
    }]

result = inf.inference_from_dicts(dicts=QA_input, rest_api_schema=True)

for x in result:
    pprint.pprint(x)

# Pure embeddings
input = [{"text": "El nombre de Berlín parece provenir de las palabras berl o birl, que en el idioma polabo que hablaban los vendos significaba tierra no cultivable o tierra deshabitada, respectivamente."}]
inf = Inferencer.load("dccuchile/bert-base-spanish-wwm-cased", task_type="embeddings")
result = inf.extract_vectors(input)

for x in result:
    pprint.pprint(x)