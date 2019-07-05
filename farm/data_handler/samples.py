from farm.data_handler.utils import get_sentence_pair

import logging

logger = logging.getLogger(__name__)


class SampleBasket:
    def __init__(self,
                 id,
                 raw,
                 samples=None):
        self.id = id
        self.raw = raw
        self.samples = samples



class Sample(object):
    """A single training/test example."""

    def __init__(self,
                 id,
                 clear_text,
                 features=None):

        self.id = id
        # "train - 1 - 1
        self.clear_text = clear_text
        self.features = features



def create_sample_one_label_one_text(raw_data, text_index, label_index, basket_id):

    # text = " ".join(raw_data[text_index:])
    text = raw_data[text_index]
    label = raw_data[label_index]

    return [Sample(id=basket_id + "-0",
                    clear_text={"text": text,
                                "label": label})]


def create_sample_ner(split_text, label, basket_id):

    text = " ".join(split_text)
    label = label

    return [Sample(id=basket_id + "-0",
                   clear_text={"text": text,
                               "label": label})]



def create_samples_lm(baskets):
    """Creates examples for Language Model Finetuning that consist of two sentences and the isNext label indicating if
     the two are subsequent sentences from one doc"""
    # sample_to_docs = lines[1]
    all_docs = [b.raw for b in baskets]
    for basket in baskets:
        doc = basket.raw
        basket.samples = []
        for idx in range(len(doc) - 1):
            id = "%s-%s" % (basket.id, idx)
            text_a, text_b, is_next_label = get_sentence_pair(doc, all_docs, idx)
            sample_in_clear_text = {
                "text_a": text_a,
                "text_b": text_b,
                "is_next_label": is_next_label,
            }
            basket.samples.append(Sample(id=id, clear_text=sample_in_clear_text))
    return baskets
