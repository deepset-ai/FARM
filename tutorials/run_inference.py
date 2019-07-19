from farm.infer import Inferencer

basic_texts = [
    {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
    {"text": "Martin Müller spielt Fussball"},
]
###################################
# Document Classification Example
###################################
model = Inferencer("save/bert-german-GNAD")
result = model.run_inference(dicts=basic_texts)
print(result)

###################################
# NER Example
###################################
model = Inferencer("save/bert-german-GNAD")
result = model.run_inference(dicts=basic_texts)
print(result)


###################################
# Question Answering Example
###################################
QA_input = [
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
    }
]

model = Inferencer("save/bert-english-squad")
result = model.run_inference(dicts=QA_input)
print(result)
