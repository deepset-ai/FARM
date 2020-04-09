from farm.infer import Inferencer
import pandas as pd
import numpy as np

model = Inferencer.load(
    "../../../bsk-customer-service/models/xlm-roberta-large-english-SemEval2017-4a",
    task_type="text_classification",
    return_class_probs=True
)

for type, file in [("chats", "../../../bsk-customer-service/data/embeddings/chat_cut/metadata.tsv"),
                   ("sentences", "../../../bsk-customer-service/data/embeddings/sentence/metadata.tsv")]:
    dataframe = pd.read_csv(file, sep="\t")

    texts = []
    for chat in dataframe["text"]:
        texts.append({"text": chat})

    predictions = model.inference_from_dicts(texts)

    sentiment_probabilities = []

    for batch in predictions:
        for instance in batch.get("predictions"):
            sentiment_probabilities.append(instance.get("probability"))

    negative_probabilities = [prob[0] for prob in sentiment_probabilities]
    positive_probabilities = [prob[2] for prob in sentiment_probabilities]

    dataframe["negative_prob_semeval"] = negative_probabilities
    dataframe["positive_prob_semeval"] = positive_probabilities

    dataframe.sort_values(by="negative_prob_semeval", inplace=True, ascending=False)
    dataframe.to_csv(f"../../../bsk-customer-service/sentiment/semeval/most_negative_{type}.tsv", sep="\t", index=False)
    dataframe.sort_values(by="positive_prob_semeval", inplace=True, ascending=False)
    dataframe.to_csv(f"../../../bsk-customer-service/sentiment/semeval/most_positive_{type}.tsv", sep="\t", index=False)

