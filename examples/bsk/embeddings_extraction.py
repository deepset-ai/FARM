import pandas as pd
from farm.infer import Inferencer
from farm.utils import set_all_seeds


def embeddings_extraction(texts, strategy="cls_token"):
    set_all_seeds(seed=42)
    batch_size = 32
    use_gpu = False
    lang_model = "bert-base-german-cased"
    # or local path:
    # lang_model = Path("../saved_models/farm-bert-base-cased-squad2")

    # Load model, tokenizer and processor directly into Inferencer
    model = Inferencer.load(lang_model, task_type="embeddings", gpu=use_gpu, batch_size=batch_size)

    # Get embeddings for input text (you can vary the strategy and layer)
    result = model.extract_vectors(dicts=texts, extraction_strategy=strategy, extraction_layer=-1)

    return result

if __name__ == "__main__":
    EMBEDDINGS_FOR = "chat"

    data = pd.read_csv("../../../bsk-customer-service/data/transformed_data.tsv", sep="\t")
    data.fillna(" ", inplace=True)
    data = data[~ data["text"].str.isspace()]

    # # extract embeddings per chat
    if EMBEDDINGS_FOR == "chat":
        data_path = "../../../bsk-customer-service/data/embeddings/chat"

        grouped = data.groupby("id")
        # extract metadata
        browser = grouped["browser"].first()
        os = grouped["os"].first()
        device = grouped["device"].first()
        login_page_title = grouped["login_page_title"].first()
        login_page_url = grouped["login_page_url"].first()

        text_series = grouped["text"].apply(" ".join)
        text_series.dropna(inplace=True)
        metadata_df = pd.concat([text_series, browser, os, device, login_page_title, login_page_url], axis=1)
        metadata_df.to_csv(data_path + "/metadata.tsv", sep="\t", index=False)

        # texts = [{"text": text} for text in text_series if (text and not text.isspace())]

        # print("Extracting Embeddings per chat (CLS-token)...")
        # embeddings = embeddings_extraction(texts)
        # embedding_list = [embedding["vec"] for embedding in embeddings]
        # # write vectors to tsv-file
        # with open(data_path + "/vecs_cls.tsv", "w") as vector_file:
        #     for embedding in embedding_list:
        #         vector_file.write("\t".join([str(value) for value in embedding]) + "\n")

        # print("Extracting Embeddings per chat (mean)...")
        # embeddings = embeddings_extraction(texts, strategy="reduce_mean")
        # embedding_list = [embedding["vec"] for embedding in embeddings]
        # # write vectors to tsv-file
        # with open(data_path + "/vecs_mean.tsv", "w") as vector_file:
        #     for embedding in embedding_list:
        #         vector_file.write("\t".join([str(value) for value in embedding]) + "\n")

    # extract embeddings per sentence
    # EMBEDDINGS_FOR = "sentence"
    # if EMBEDDINGS_FOR == "sentence":
    #     data_path = "../../../bsk-customer-service/data/embeddings/most_informative_sentence"

    #     # empty string error when using all sentences,
    #     # might be due to not supported chars in strings
    #     # -> only use strings with len > 9
    #     data = data[data.text.str.len() > 9]
    #     data.to_csv(data_path + "/metadata2.tsv", sep="\t", index=False)

    #     texts = [{"text" : text} for text in data.text if (text and not text.isspace())]

    #     print("Extracting Embeddings per sentence (CLS-token)...")
    #     embeddings = embeddings_extraction(texts)
    #     embedding_list = [embedding["vec"] for embedding in embeddings]
    #     # write vectors to tsv-file
    #     with open(data_path + "/vecs_cls2.tsv", "w") as vector_file:
    #         for embedding in embedding_list:
    #             vector_file.write("\t".join([str(value) for value in embedding]) + "\n")

    #     print("Extracting Embeddings per sentence (mean)...")
    #     embeddings = embeddings_extraction(texts, strategy="reduce_mean")
    #     embedding_list = [embedding["vec"] for embedding in embeddings]
    #     # write vectors to tsv-file
    #     with open(data_path + "/vecs_mean2.tsv", "w") as vector_file:
    #         for embedding in embedding_list:
    #             vector_file.write("\t".join([str(value) for value in embedding]) + "\n")
