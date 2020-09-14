from transformers.tokenization_bert import whitespace_tokenize
from farm.visual.ascii.images import SAMPLE

import logging

logger = logging.getLogger(__name__)


class SampleBasket:
    """ An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(self, id_internal: str, raw: dict, id_external=None, samples=None):
        """
        :param id: A unique identifying id. Used for identification within FARM.
        :type id: str
        :param external_id: Used for identification outside of FARM. E.g. if another framework wants to pass along its own id with the results.
        :type external_id: str
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :type raw: dict
        :param samples: An optional list of Samples used to populate the basket at initialization.
        :type samples: Sample
        """
        self.id_internal = id_internal
        self.id_external = id_external
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
            if len(clear_text_str) > 10000:
                clear_text_str = clear_text_str[:10_000] + f"\nTHE REST IS TOO LONG TO DISPLAY. " \
                                                           f"Remaining chars :{len(clear_text_str)-10_000}"
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
            if len(tokenized_str) > 10000:
                tokenized_str = tokenized_str[:10_000] + f"\nTHE REST IS TOO LONG TO DISPLAY. " \
                                                         f"Remaining chars: {len(tokenized_str)-10_000}"
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


def create_sample_one_label_one_text(raw_data, text_index, label_index, basket_id):

    # text = " ".join(raw_data[text_index:])
    text = raw_data[text_index]
    label = raw_data[label_index]

    return [Sample(id=basket_id + "-0", clear_text={"text": text, "label": label})]


def create_sample_ner(split_text, label, basket_id):

    text = " ".join(split_text)
    label = label

    return [Sample(id=basket_id + "-0", clear_text={"text": text, "label": label})]


def process_answers(answers, doc_offsets, passage_start_c, passage_start_t):
    """TODO Write Comment"""
    answers_clear = []
    answers_tokenized = []
    for answer in answers:
        # This section calculates start and end relative to document
        answer_text = answer["text"]
        answer_len_c = len(answer_text)
        answer_start_c = answer["offset"]
        answer_end_c = answer_start_c + answer_len_c - 1
        answer_start_t = offset_to_token_idx(doc_offsets, answer_start_c)
        answer_end_t = offset_to_token_idx(doc_offsets, answer_end_c)

        # TODO: Perform check that answer can be recovered from document?

        # This section converts start and end so that they are relative to the passage
        # TODO: Is this actually necessary on character level?
        answer_start_c -= passage_start_c
        answer_end_c -= passage_start_c
        answer_start_t -= passage_start_t
        answer_end_t -= passage_start_t

        curr_answer_clear = {"text": answer_text,
                             "start_c": answer_start_c,
                             "end_c": answer_end_c}
        curr_answer_tokenized = {"start_t": answer_start_t,
                                 "end_t": answer_end_t,
                                 "answer_type": answer["answer_type"]}

        answers_clear.append(curr_answer_clear)
        answers_tokenized.append(curr_answer_tokenized)
    return answers_clear, answers_tokenized


def create_samples_qa(dictionary, max_query_len, max_seq_len, doc_stride, n_special_tokens):
    """
    This method will split question-document pairs from the SampleBasket into question-passage pairs which will
    each form one sample. The "t" and "c" in variables stand for token and character respectively.
    """

    # Initialize some basic variables
    # is_training = check_if_training(dictionary)
    question_tokens = dictionary["question_tokens"][:max_query_len]
    question_len_t = len(question_tokens)
    question_offsets = dictionary["question_offsets"]
    doc_tokens = dictionary["document_tokens"]
    doc_offsets = dictionary["document_offsets"]
    doc_text = dictionary["document_text"]
    doc_start_of_word = dictionary["document_start_of_word"]
    samples = []

    # Calculate the number of tokens that can be reserved for the passage. This is calculated by considering
    # the max_seq_len, the number of tokens in the question and the number of special tokens that will be added
    # when the question and passage are joined (e.g. [CLS] and [SEP])
    passage_len_t = max_seq_len - question_len_t - n_special_tokens

    # Perform chunking of document into passages. The sliding window moves in steps of doc_stride.
    # passage_spans is a list of dictionaries where each defines the start and end of each passage
    # on both token and character level
    passage_spans = chunk_into_passages(doc_offsets,
                                        doc_stride,
                                        passage_len_t,
                                        doc_text)
    for passage_span in passage_spans:
        # Unpack each variable in the dictionary. The "_t" and "_c" indicate
        # whether the index is on the token or character level
        passage_start_t = passage_span["passage_start_t"]
        passage_end_t = passage_span["passage_end_t"]
        passage_start_c = passage_span["passage_start_c"]
        passage_end_c = passage_span["passage_end_c"]
        passage_id = passage_span["passage_id"]

        # passage_offsets will be relative to the start of the passage (i.e. they will start at 0)
        # TODO: Is passage offsets actually needed? At this point, maybe we only care about token level
        passage_offsets = doc_offsets[passage_start_t: passage_end_t]
        passage_start_of_word = doc_start_of_word[passage_start_t: passage_end_t]
        passage_offsets = [x - passage_offsets[0] for x in passage_offsets]
        passage_tokens = doc_tokens[passage_start_t: passage_end_t]
        passage_text = dictionary["document_text"][passage_start_c: passage_end_c]

        # Deal with the potentially many answers (e.g. Squad or NQ dev set)
        answers_clear, answers_tokenized = process_answers(dictionary["answers"],
                                                           doc_offsets,
                                                           passage_start_c,
                                                           passage_start_t)

        clear_text = {"passage_text": passage_text,
                      "question_text": dictionary["question_text"],
                      "passage_id": passage_id,
                      "answers": answers_clear}
        tokenized = {"passage_start_t": passage_start_t,
                     "passage_tokens": passage_tokens,
                     "passage_offsets": passage_offsets,
                     "passage_start_of_word": passage_start_of_word,
                     "question_tokens": question_tokens,
                     "question_offsets": question_offsets,
                     "question_start_of_word": dictionary["question_start_of_word"][:max_query_len],
                     "answers": answers_tokenized,
                     "document_offsets": doc_offsets}   # So that to_doc_preds can access them
        samples.append(Sample(id=passage_id,
                              clear_text=clear_text,
                              tokenized=tokenized))
    return samples


def chunk_into_passages(doc_offsets,
                        doc_stride,
                        passage_len_t,
                        doc_text):
    """ Returns a list of dictionaries which each describe the start, end and id of a passage
    that is formed when chunking a document using a sliding window approach. """

    assert doc_stride < passage_len_t, "doc_stride is longer than passage_len_t. This means that there will be gaps " \
                                       "as the passage windows slide, causing the model to skip over parts of the document. "\
                                       "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384) "

    passage_spans = []
    passage_id = 0
    doc_len_t = len(doc_offsets)
    while True:
        passage_start_t = passage_id * doc_stride
        passage_end_t = passage_start_t + passage_len_t
        passage_start_c = doc_offsets[passage_start_t]

        # If passage_end_t points to the last token in the passage, define passage_end_c as the length of the document
        if passage_end_t >= doc_len_t - 1:
            passage_end_c = len(doc_text)

        # Get document text up to the first token that is outside the passage. Strip of whitespace.
        # Use the length of this text as the passage_end_c
        else:
            end_ch_idx = doc_offsets[passage_end_t + 1]
            raw_passage_text = doc_text[:end_ch_idx]
            passage_end_c = len(raw_passage_text.strip())

        passage_span = {"passage_start_t": passage_start_t,
                        "passage_end_t": passage_end_t,
                        "passage_start_c": passage_start_c,
                        "passage_end_c": passage_end_c,
                        "passage_id": passage_id}
        passage_spans.append(passage_span)
        passage_id += 1
        # If the end idx is greater than or equal to the length of the passage
        if passage_end_t >= doc_len_t:
            break
    return passage_spans


def offset_to_token_idx(token_offsets, ch_idx):
    """ Returns the idx of the token at the given character idx"""
    n_tokens = len(token_offsets)
    for i in range(n_tokens):
        if (i + 1 == n_tokens) or (token_offsets[i] <= ch_idx < token_offsets[i + 1]):
            return i
