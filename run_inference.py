from farm.infer import Inferencer

load_dir = "save/bert-german-CONLL2003"
# load_dir = "save/bert-german-GNAD"
# load_dir = "save/bert-german-GermEval18Fine"

# raw_data = [
#     [None, "This is one sentence"],
#     [None, "This is another sentence"],
#     [None, "Let's try a third sentence"],
# ]

# raw_data = [
#     ["This is one sentence", None],
#     ["This is another sentence", None],
#     ["This is a third sentence", None],
# ]

dicts = [
    {"text": "Schartau sagte dem Tagesspiegel."},
    {"text": "Martin spielt Fussball"},
]
model = Inferencer(load_dir)

# samples = [{"texts": "Barack Obama was a president of the US"}]
# raw_data = [[sample["texts"], None] for sample in samples]
result = model.run_inference(dicts=dicts)

print(result)
print(model.language)
