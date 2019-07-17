from farm.infer import Inferencer

# load_dir = "save/CONLL2003"
load_dir = "save/GNAD"

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

dicts = [{"text": "This is one sentence"}, {"text": "This is another sentence"}]
model = Inferencer(load_dir)

# samples = [{"texts": "Barack Obama was a president of the US"}]
# raw_data = [[sample["texts"], None] for sample in samples]
result = model.run_inference(raw_data=dicts)

print(result)
print(model.language)
