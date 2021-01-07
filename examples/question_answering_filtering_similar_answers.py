from farm.infer import QAInferencer
from pprint import pprint

QA_input = [
        {
            "questions": ["“In what country lies the Normandy?”"],
            "text":  """The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\")
            raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia.
            The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. Weird things happen in Normandy, France."""
        }]

infer = QAInferencer.load("deepset/roberta-base-squad2", task_type="question_answering", gpu=True)
infer.model.prediction_heads[0].n_best = 5
infer.model.prediction_heads[0].n_best_per_sample = 5

# To filter duplicates, each pair of answers where the start indices or end indices differ by 5 or less are handled as duplicates with the following parameter setting.
# Setting this parameter to 0 filters exact duplicates: pairs of answers that have the same start indices or end indices.
# Setting this parameter to -1 turns off duplicate removal (default).
infer.model.prediction_heads[0].filter_range = 5

result = infer.inference_from_dicts(dicts=QA_input, return_json=False)

for r in result:
    pprint(r.to_json())
