from farm.infer import Inferencer

# load_dir = "save/bert-german-CONLL2003"
# load_dir = "save/bert-german-GNAD"
# load_dir = "save/bert-german-GermEval18Fine"
load_dir = "save/qa_model_1"


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

QA_dicts = [
    {
        "paragraphs": [
            {
                "qas": [
                    {
                        "question": "When did Beyonce start becoming popular?",
                        "id": "123",
                    }
                ],
                "context": 'Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            }
        ]
    },
    {
        "paragraphs": [
            {
                "qas": [
                    {
                        "question": "After her second solo album, what other entertainment venture did Beyonce explore??",
                        "id": "124",
                    }
                ],
                "context": 'Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            }
        ]
    },
]
model = Inferencer(load_dir)
# samples = [{"texts": "Barack Obama was a president of the US"}]
# raw_data = [[sample["texts"], None] for sample in samples]
result = model.run_inference(dicts=QA_dicts)
print(result)
print(model.language)
