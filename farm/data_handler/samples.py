from farm.data_handler.utils import get_sentence_pair
from transformers.tokenization_bert import whitespace_tokenize
from farm.modeling.tokenization import tokenize_with_metadata
from farm.visual.ascii.images import SAMPLE
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class SampleBasket:
    """ An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(self, id: str, raw: dict, samples=None):
        """
        :param id: A unique identifying id.
        :type id: str
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :type raw: dict
        :param samples: An optional list of Samples used to populate the basket at initialization.
        """
        self.id = id
        self.raw = raw
        self.samples = samples


class Sample(object):
    """A single training/test sample. This should contain the input and the label. Is initialized with
    the human readable clear_text. Over the course of data preprocessing, this object is populated
    with tokenized and featurized versions of the data."""

    def __init__(self, id, clear_text, tokenized=None, features=None):
        """
        :param id: The unique id of the sample
        :type id: str
        :param clear_text: A dictionary containing various human readable fields (e.g. text, label).
        :type clear_text: dict
        :param tokenized: A dictionary containing the tokenized version of clear text plus helpful meta data: offsets (start position of each token in the original text) and start_of_word (boolean if a token is the first one of a word).
        :type tokenized: dict
        :param features: A dictionary containing features in a vectorized format needed by the model to process this sample.
        :type features: dict

        """
        self.id = id
        self.clear_text = clear_text
        self.features = features
        self.tokenized = tokenized

    def __str__(self):
        if self.clear_text:
            clear_text_str = "\n \t".join(
                [k + ": " + str(v) for k, v in self.clear_text.items()]
            )
        else:
            clear_text_str = "None"
        if self.features:
            if isinstance(self.features, list):
                features = self.features[0]
            else:
                features = self.features
            feature_str = "\n \t".join([k + ": " + str(v) for k, v in features.items()])
        else:
            feature_str = "None"

        if self.tokenized:
            tokenized_str = "\n \t".join(
                [k + ": " + str(v) for k, v in self.tokenized.items()]
            )
        else:
            tokenized_str = "None"
        s = (
            f"\n{SAMPLE}\n"
            f"ID: {self.id}\n"
            f"Clear Text: \n \t{clear_text_str}\n"
            f"Tokenized: \n \t{tokenized_str}\n"
            f"Features: \n \t{feature_str}\n"
            "_____________________________________________________"
        )
        return s


class Squad_cleartext:
    def __init__(
        self,
        qas_id,
        question_text,
        doc_tokens,
        orig_answer_text,
        start_position,
        end_position,
        is_impossible,
    ):

        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def create_sample_one_label_one_text(raw_data, text_index, label_index, basket_id):

    # text = " ".join(raw_data[text_index:])
    text = raw_data[text_index]
    label = raw_data[label_index]

    return [Sample(id=basket_id + "-0", clear_text={"text": text, "label": label})]


def create_sample_ner(split_text, label, basket_id):

    text = " ".join(split_text)
    label = label

    return [Sample(id=basket_id + "-0", clear_text={"text": text, "label": label})]


def create_samples_sentence_pairs(baskets, tokenizer, max_seq_len):
    """Creates examples for Language Model Finetuning that consist of two sentences and the isNext label indicating if
     the two are subsequent sentences from one doc"""
    all_docs = [b.raw["doc"] for b in baskets]
    for basket in tqdm(baskets):
        doc = basket.raw["doc"]
        basket.samples = []
        for idx in range(len(doc) - 1):
            id = "%s-%s" % (basket.id, idx)
            text_a, text_b, is_next_label = get_sentence_pair(doc, all_docs, idx)
            sample_in_clear_text = {
                "text_a": text_a,
                "text_b": text_b,
                "is_next_label": is_next_label,
            }
            tokenized = {}
            tokenized["text_a"] = tokenize_with_metadata(text_a, tokenizer, max_seq_len)
            tokenized["text_b"] = tokenize_with_metadata(text_b, tokenizer, max_seq_len)
            basket.samples.append(Sample(id=id, clear_text=sample_in_clear_text, tokenized=tokenized))
    return baskets


def create_samples_squad(entry):
    """Read a SQuAD json file into a list of SquadExample."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    try:
        _ = entry["paragraphs"][0]["qas"][0]["is_impossible"]
        is_training = True
    except KeyError:
        is_training = False

    examples = []
    num_examples = 1
    for paragraph in entry["paragraphs"]:
        paragraph_text = paragraph["context"]

        char_to_word_offset = []
        doc_tokens = paragraph_text.split(" ")
        for i, t in enumerate(doc_tokens):
            char_to_word_offset.extend([i] * (len(t) + 1))
        char_to_word_offset = char_to_word_offset[:-1]  # cut off last added whitespace

        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            if is_training:
                is_impossible = qa["is_impossible"]
                # TODO check how to transform dev set with multiple possible answers, for now take only 1 answer
                # if (len(qa["answers"]) != 1) and (not is_impossible):
                #     raise ValueError(
                #         "For training, each question should have exactly 1 answer."
                #     )
                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[
                        answer_offset + answer_length - 1
                    ]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position : (end_position + 1)]
                    )
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text)
                    )
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning(
                            "Could not find answer: '%s' vs. '%s'",
                            actual_text,
                            cleaned_answer_text,
                        )
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            clear_text = {}
            clear_text["qas_id"] = qas_id
            clear_text["question_text"] = question_text
            clear_text["doc_tokens"] = doc_tokens
            clear_text["orig_answer_text"] = orig_answer_text
            clear_text["start_position"] = start_position
            clear_text["end_position"] = end_position
            clear_text["is_impossible"] = is_impossible
            clear_text["is_training"] = is_training
            example = Sample(
                id=None, clear_text=clear_text, features=None, tokenized=None
            )
            num_examples += 1
            examples.append(example)
    return examples
