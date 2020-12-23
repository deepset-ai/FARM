"""
Contains functions that make Natural Question work with old Processor code
These functions should be deprecated soon
"""


import logging
import re
import numpy as np

from farm.data_handler.samples import Sample

logger = logging.getLogger(__name__)



def sample_to_features_qa_Natural_Questions(sample, tokenizer, max_seq_len, sp_toks_start, sp_toks_mid, sp_toks_end,
                          answer_type_list=None, max_answers=6):
    """ Prepares data for processing by the model. Supports cases where there are
    multiple answers for the one question/document pair. max_answers is by default set to 6 since
    that is the most number of answers in the squad2.0 dev set.

    :param sample: A Sample object that contains one question / passage pair
    :type sample: Sample
    :param tokenizer: A Tokenizer object
    :type tokenizer: Tokenizer
    :param max_seq_len: The maximum sequence length
    :type max_seq_len: int
    :param sp_toks_start: The number of special tokens that come before the question tokens
    :type sp_toks_start: int
    :param sp_toks_mid: The number of special tokens that come between the question and passage tokens
    :type sp_toks_mid: int
    :param answer_type_list: A list of all the answer types that can be expected e.g. ["no_answer", "span", "yes", "no"] for Natural Questions
    :type answer_type_list: List[str]
    :param max_answers: The maximum number of answer annotations for a sample (In SQuAD, this is 6 hence the default)
    :type max_answers: int
    :return: dict (keys: [input_ids, padding_mask, segment_ids, answer_type_ids, passage_start_t, start_of_word, labels, id, seq_2_start_2])
    """

    # Initialize some basic variables
    question_tokens = sample.tokenized["question_tokens"]
    question_start_of_word = sample.tokenized["question_start_of_word"]
    question_len_t = len(question_tokens)
    passage_start_t = sample.tokenized["passage_start_t"]
    passage_tokens = sample.tokenized["passage_tokens"]
    passage_start_of_word = sample.tokenized["passage_start_of_word"]
    passage_len_t = len(passage_tokens)
    answers = sample.tokenized["answers"]
    sample_id = [int(x) for x in sample.id.split("-")]

    # Generates a numpy array of shape (max_answers, 2) where (i, 2) indexes into the start and end indices
    # of the ith answer. The array is filled with -1 since the number of answers is often less than max_answers
    # no answer labels are represented by (0,0)
    labels, answer_types = generate_labels(answers,
                                           passage_len_t,
                                           question_len_t,
                                           max_answers,
                                           sp_toks_start,
                                           sp_toks_mid,
                                           answer_type_list)

    # Generate a start of word vector for the full sequence (i.e. question + answer + special tokens).
    # This will allow us to perform evaluation during training without clear text.
    # Note that in the current implementation, special tokens do not count as start of word.
    start_of_word = combine_vecs(question_start_of_word, passage_start_of_word, tokenizer, spec_tok_val=0)

    # Combines question_tokens and passage_tokens (str) into a single encoded vector of token indices (int)
    # called input_ids. This encoded vector also contains special tokens (e.g. [CLS]). It will have length =
    # (question_len_t + passage_len_t + n_special_tokens). This may be less than max_seq_len but will not be greater
    # than max_seq_len since truncation was already performed when the document was chunked into passages
    # (c.f. create_samples_squad() )

    if tokenizer.is_fast:
        # Detokenize input as fast tokenizer can't handle tokenized input
        question_tokens = " ".join(question_tokens)
        question_tokens = re.sub(r"(^|\s)(##)", "", question_tokens)
        passage_tokens = " ".join(passage_tokens)
        passage_tokens = re.sub(r"(^|\s)(##)", "", passage_tokens)

        encoded = tokenizer(text=question_tokens,
                            text_pair=passage_tokens,
                            add_special_tokens=True,
                            return_special_tokens_mask=True,
                            return_token_type_ids=True)

        n_tokens_encoded = len(encoded["input_ids"]) - encoded["special_tokens_mask"].count(1)
        n_tokens_with_metadata = len(sample.tokenized["question_tokens"]) + len(sample.tokenized["passage_tokens"])

        if n_tokens_encoded != n_tokens_with_metadata:
            tokens_encoded = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            logger.error(f"FastTokenizer encoded sample to {n_tokens_encoded} tokens,"
                         f" while the previous tokenize_with_metadata produced {n_tokens_with_metadata} tokens. \n"
                         f"Further processing is likely to be wrong.\n"
                         f"FastTokenizer: {tokens_encoded} \n"
                         f"tokenize_with_metadata: {sample.tokenized['question_tokens'] + sample.tokenized['passage_tokens']}")
    else:
        encoded = tokenizer.encode_plus(text=sample.tokenized["question_tokens"],
                                        text_pair=sample.tokenized["passage_tokens"],
                                        add_special_tokens=True,
                                        truncation=False,
                                        truncation_strategy='do_not_truncate',
                                        return_token_type_ids=True,
                                        return_tensors=None)

    input_ids = encoded["input_ids"]
    segment_ids = encoded["token_type_ids"]

    # seq_2_start_t is the index of the first token in the second text sequence (e.g. passage)
    if tokenizer.__class__.__name__ in ["RobertaTokenizer", "XLMRobertaTokenizer"]:
        seq_2_start_t = get_roberta_seq_2_start(input_ids)
    elif tokenizer.__class__.__name__ == "CamembertTokenizer":
        seq_2_start_t = get_camembert_seq_2_start(input_ids)
    else:
        seq_2_start_t = segment_ids.index(1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # The passage mask has 1 for tokens that are valid start or ends for QA spans.
    # 0s are assigned to question tokens, mid special tokens, end special tokens and padding
    # Note that start special tokens are assigned 1 since they can be chosen for a no_answer prediction
    span_mask = [1] * sp_toks_start
    span_mask += [0] * question_len_t
    span_mask += [0] * sp_toks_mid
    span_mask += [1] * passage_len_t
    span_mask += [0] * sp_toks_end

    # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
    pad_idx = tokenizer.pad_token_id
    padding = [pad_idx] * (max_seq_len - len(input_ids))
    zero_padding = [0] * (max_seq_len - len(input_ids))

    input_ids += padding
    padding_mask += zero_padding
    segment_ids += zero_padding
    start_of_word += zero_padding
    span_mask += zero_padding

    # The XLM-Roberta tokenizer generates a segment_ids vector that separates the first sequence from the second.
    # However, when this is passed in to the forward fn of the Roberta model, it throws an error since
    # Roberta has only a single token embedding (!!!). To get around this, we want to have a segment_ids
    # vec that is only 0s
    if tokenizer.__class__.__name__ in ["XLMRobertaTokenizer", "RobertaTokenizer"]:
        segment_ids = list(np.zeros_like(segment_ids))

    # The first of the labels will be used in train, and the full array will be used in eval.
    # start of word and spec_tok_mask are not actually needed by model.forward() but are needed for model.formatted_preds()
    # passage_start_t is index of passage's first token relative to document
    feature_dict = {"input_ids": input_ids,
                    "padding_mask": padding_mask,
                    "segment_ids": segment_ids,
                    "answer_type_ids": answer_types,
                    "passage_start_t": passage_start_t,
                    "start_of_word": start_of_word,
                    "labels": labels,
                    "id": sample_id,
                    "seq_2_start_t": seq_2_start_t,
                    "span_mask": span_mask}
    return [feature_dict]

def generate_labels(answers, passage_len_t, question_len_t, max_answers,
                    sp_toks_start, sp_toks_mid, answer_type_list=None):
    """
    Creates QA label vector for each answer in answers. The labels are the index of the start and end token
    relative to the passage. They are contained in an array of size (max_answers, 2).
    -1 is used to fill array since there the number of answers is often less than max_answers.
    The index values take in to consideration the question tokens, and also special tokens such as [CLS].
    When the answer is not fully contained in the passage, or the question
    is impossible to answer, the start_idx and end_idx are 0 i.e. start and end are on the very first token
    (in most models, this is the [CLS] token). Note that in our implementation NQ has 4 answer types
    ["no_answer", "yes", "no", "span"] and this is what answer_type_list should look like"""

    # Note here that label_idxs get passed to the QuestionAnsweringHead and answer_types get passed to the text
    # classification head. label_idxs may contain multiple start, end labels since SQuAD dev and test sets
    # can have multiple annotations. By contrast, Natural Questions only has one annotation per sample hence
    # why answer_types is only of length 1
    label_idxs = np.full((max_answers, 2), fill_value=-1)
    answer_types = np.asarray([-1])
    answer_str = ""

    # If there are no answers
    if len(answers) == 0:
        label_idxs[0, :] = 0
        answer_types[:] = 0
        return label_idxs, answer_types

    # Iterate over the answers for the one sample
    for i, answer in enumerate(answers):
        start_idx = answer["start_t"]
        end_idx = answer["end_t"]

        # Check that the start and end are contained within this passage
        if answer_in_passage(start_idx, end_idx, passage_len_t):
            label_idxs[i][0] = sp_toks_start + question_len_t + sp_toks_mid + start_idx
            label_idxs[i][1] = sp_toks_start + question_len_t + sp_toks_mid + end_idx
            answer_str = answer["answer_type"]
        # If the start or end of the span answer is outside the passage, treat passage as no_answer
        else:
            label_idxs[i][0] = 0
            label_idxs[i][1] = 0
            answer_str = "no_answer"

    if answer_type_list:
        answer_types[0] = answer_type_list.index(answer_str)

    return label_idxs, answer_types



def combine_vecs(question_vec, passage_vec, tokenizer, spec_tok_val=-1):
    """ Combine a question_vec and passage_vec in a style that is appropriate to the model. Will add slots in
    the returned vector for special tokens like [CLS] where the value is determine by spec_tok_val."""

    # Join question_label_vec and passage_label_vec and add slots for special tokens
    vec = tokenizer.build_inputs_with_special_tokens(token_ids_0=question_vec,
                                                     token_ids_1=passage_vec)
    if tokenizer.is_fast:
        spec_toks_mask = tokenizer.get_special_tokens_mask(token_ids_0=vec,
                                                           already_has_special_tokens=True)
    else:
        spec_toks_mask = tokenizer.get_special_tokens_mask(token_ids_0=question_vec,
                                                           token_ids_1=passage_vec)

    # If a value in vec corresponds to a special token, it will be replaced with spec_tok_val
    combined = [v if not special_token else spec_tok_val for v, special_token in zip(vec, spec_toks_mask)]

    return combined


def answer_in_passage(start_idx, end_idx, passage_len):
    if passage_len > start_idx >= 0 and passage_len > end_idx > 0:
        return True
    return False

def get_roberta_seq_2_start(input_ids):
    # This commit (https://github.com/huggingface/transformers/commit/dfe012ad9d6b6f0c9d30bc508b9f1e4c42280c07)from
    # huggingface transformers now means that RobertaTokenizer.encode_plus returns only zeros in token_type_ids. Therefore, we need
    # another way to infer the start of the second input sequence in RoBERTa. Roberta input sequences have the following
    # format: <s> P1 </s> </s> P2 </s>
    # <s> has index 0 and </s> has index 2. To find the beginning of the second sequence, this function first finds
    # the index of the second </s>
    first_backslash_s = input_ids.index(2)
    second_backslash_s = input_ids.index(2, first_backslash_s + 1)
    return second_backslash_s + 1

def get_camembert_seq_2_start(input_ids):
    # CamembertTokenizer.encode_plus returns only zeros in token_type_ids (same as RobertaTokenizer).
    # This is another way to find the start of the second sequence (following get_roberta_seq_2_start)
    # Camembert input sequences have the following
    # format: <s> P1 </s> </s> P2 </s>
    # <s> has index 5 and </s> has index 6. To find the beginning of the second sequence, this function first finds
    # the index of the second </s>
    first_backslash_s = input_ids.index(6)
    second_backslash_s = input_ids.index(6, first_backslash_s + 1)
    return second_backslash_s + 1

def create_samples_qa_Natural_Question(dictionary, max_query_len, max_seq_len, doc_stride, n_special_tokens):
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

def process_answers(answers, doc_offsets, passage_start_c, passage_start_t):
    """TODO Write Comment"""
    answers_clear = []
    answers_tokenized = []
    for answer in answers:
        # This section calculates start and end relative to document
        answer_text = answer["text"]
        answer_len_c = len(answer_text)
        if "offset" in answer:
            answer_start_c = answer["offset"]
        else:
            answer_start_c = answer["answer_start"]
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
                                 "answer_type": answer.get("answer_type","span")}

        answers_clear.append(curr_answer_clear)
        answers_tokenized.append(curr_answer_tokenized)
    return answers_clear, answers_tokenized


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

def convert_qa_input_dict(infer_dict):
    """ Input dictionaries in QA can either have ["context", "qas"] (internal format) as keys or
    ["text", "questions"] (api format). This function converts the latter into the former. It also converts the
    is_impossible field to answer_type so that NQ and SQuAD dicts have the same format.
    """
    try:
        # Check if infer_dict is already in internal json format
        if "context" in infer_dict and "qas" in infer_dict:
            return infer_dict
        # converts dicts from inference mode to data structure used in FARM
        questions = infer_dict["questions"]
        text = infer_dict["text"]
        uid = infer_dict.get("id", None)
        qas = [{"question": q,
                "id": uid,
                "answers": [],
                "answer_type": None} for i, q in enumerate(questions)]
        converted = {"qas": qas,
                     "context": text}
        return converted
    except KeyError:
        raise Exception("Input does not have the expected format")