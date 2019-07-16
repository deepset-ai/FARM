from farm.inference import Inferencer

load_dir = "save/CONLL2003"
# load_dir = "save/GNAD"

# raw_data = [
#     [None, "This is one sentence"],
#     [None, "This is another sentence"],
#     [None, "Let's try a third sentence"],
# ]

raw_data = [
    ["This is one sentence", None],
    ["This is another sentence", None],
    ["This is a third sentence", None],
]


model = Inferencer(load_dir)
# samples = [{"texts": "Barack Obama was a president of the US"}]
# raw_data = [[sample["texts"], None] for sample in samples]
result = model.run_inference(raw_data=raw_data)

print(result)
print(model.language)
