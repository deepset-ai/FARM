"""
Contains functions that turn readable clear text input into dictionaries of features
"""


import logging
import collections
from dotmap import DotMap
import numpy as np

from farm.data_handler.samples import Sample
from farm.data_handler.utils import (
    expand_labels,
    pad,
    mask_random_words,
)
from farm.modeling.tokenization import insert_at_special_tokens_pos

logger = logging.getLogger(__name__)


def sample_to_features_text(
    sample, tasks, max_seq_len, tokenizer
):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    :rtype: list
    """

    #TODO It might be cleaner to adjust the data structure in sample.tokenized
    # Verify if this current quickfix really works for pairs
    tokens_a = sample.tokenized["tokens"]
    tokens_b = sample.tokenized.get("tokens_b", None)

    inputs = tokenizer.encode_plus(
        tokens_a,
        tokens_b,
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation_strategy='do_not_truncate' # We've already truncated our tokens before
    )

    input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)


    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feat_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
    }

    # Add Labels for different tasks
    for task_name, task in tasks.items():
        try:
            label_name = task["label_name"]
            label_raw = sample.clear_text[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                try:
                    label_ids = [label_list.index(label_raw)]
                except ValueError as e:
                    raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            elif task["task_type"] == "regression":
                label_ids = [float(label_raw)]
            else:
                raise ValueError(task["task_type"])
        except KeyError:
            # For inference mode we don't expect labels
            label_ids = None
        if label_ids is not None:
            feat_dict[task["label_tensor_name"]] = label_ids
    return [feat_dict]


def samples_to_features_ner(
    sample,
    tasks,
    max_seq_len,
    tokenizer,
    non_initial_token="X",
    **kwargs
):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by an NER model.

    :param sample: Sample object that contains human readable text and label fields from a single NER data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :param non_initial_token: Token that is inserted into the label sequence in positions where there is a
                              non-word-initial token. This is done since the default NER performs prediction
                              only on word initial tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask", "segment_ids", "initial_mask"
             (also "label_ids" if not in inference mode). The values are lists containing those features.
    :rtype: list
    """

    tokens = sample.tokenized["tokens"]
    inputs = tokenizer.encode_plus(text=tokens,
                                   text_pair=None,
                                   add_special_tokens=True,
                                   max_length=max_seq_len,
                                   truncation_strategy='do_not_truncate', # We've already truncated our tokens before
                                   return_special_tokens_mask=True

    )

    input_ids, segment_ids, special_tokens_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["special_tokens_mask"]

    # We construct a mask to identify the first token of a word. We will later only use them for predicting entities.
    # Special tokens don't count as initial tokens => we add 0 at the positions of special tokens
    # For BERT we add a 0 in the start and end (for CLS and SEP)
    initial_mask = [int(x) for x in sample.tokenized["start_of_word"]]
    initial_mask = insert_at_special_tokens_pos(initial_mask, special_tokens_mask, insert_element=0)
    assert len(initial_mask) == len(input_ids)

    for task_name, task in tasks.items():
        try:
            label_list = task["label_list"]
            label_name = task["label_name"]
            label_tensor_name = task["label_tensor_name"]
            labels_word = sample.clear_text[label_name]
            labels_token = expand_labels(labels_word, initial_mask, non_initial_token)
            # labels_token = add_cls_sep(labels_token, cls_token, sep_token)
            label_ids = [label_list.index(lt) for lt in labels_token]
        except ValueError:
            label_ids = None
            problematic_labels = set(labels_token).difference(set(label_list))
            logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                           f"\nWe found a problem with labels {str(problematic_labels)}")
        except KeyError:
            # For inference mode we don't expect labels
            label_ids = None
            logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                           "\nIf your are running in *inference* mode: Don't worry!"
                           "\nIf you are running in *training* mode: Verify you are supplying a proper label list to your processor and check that labels in input data are correct.")

        # This mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)

        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4" for padding token_type_ids
        if tokenizer.__class__.__name__ == "XLNetTokenizer":
            pad_on_left = True
            segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)
        initial_mask = pad(initial_mask, max_seq_len, 0, pad_on_left=pad_on_left)
        if label_ids:
            label_ids = pad(label_ids, max_seq_len, 0, pad_on_left=pad_on_left)

        feature_dict = {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
            "initial_mask": initial_mask,
        }

        if label_ids:
            feature_dict[label_tensor_name] = label_ids

    return [feature_dict]


def samples_to_features_bert_lm(sample, max_seq_len, tokenizer, next_sent_pred=True):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :type sample: Sample
    :param max_seq_len: Maximum length of sequence.
    :type max_seq_len: int
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    if next_sent_pred:
        tokens_a = sample.tokenized["text_a"]["tokens"]
        tokens_b = sample.tokenized["text_b"]["tokens"]

        # mask random words
        tokens_a, t1_label = mask_random_words(tokens_a, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_a"]["start_of_word"])

        tokens_b, t2_label = mask_random_words(tokens_b, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_b"]["start_of_word"])
        # convert lm labels to ids
        t1_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t1_label]
        t2_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t2_label]
        lm_label_ids = t1_label_ids + t2_label_ids

        # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
        if sample.clear_text["nextsentence_label"]:
            is_next_label_id = [0]
        else:
            is_next_label_id = [1]
    else:
        tokens_a = sample.tokenized["text_a"]["tokens"]
        tokens_b = None
        tokens_a, t1_label = mask_random_words(tokens_a, tokenizer.vocab,
                                               token_groups=sample.tokenized["text_a"]["start_of_word"])
        # convert lm labels to ids
        lm_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t1_label]

    # encode string tokens to input_ids and add special tokens
    inputs = tokenizer.encode_plus(text=tokens_a,
                                   text_pair=tokens_b,
                                   add_special_tokens=True,
                                   max_length=max_seq_len,
                                   truncation_strategy='do_not_truncate',
                                   # We've already truncated our tokens before
                                   return_special_tokens_mask=True
                                   )

    input_ids, segment_ids, special_tokens_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
        "special_tokens_mask"]

    # account for special tokens (CLS, SEP, SEP..) in lm_label_ids
    lm_label_ids = insert_at_special_tokens_pos(lm_label_ids, special_tokens_mask, insert_element=-1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4" for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)
    lm_label_ids = pad(lm_label_ids, max_seq_len, -1, pad_on_left=pad_on_left)

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "lm_label_ids": lm_label_ids,
    }

    if next_sent_pred:
        feature_dict["nextsentence_label_ids"] = is_next_label_id

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    return [feature_dict]


def sample_to_features_squad(sample, tokenizer, max_seq_len, max_answers=6):
    """ Prepares data for processing by the model. Supports cases where there are
    multiple answers for the one question/document pair. max_answers is by default set to 6 since
    that is the most number of answers in the squad2.0 dev set."""

    # Initialize some basic variables
    is_impossible = sample.clear_text["is_impossible"]
    question_tokens = sample.tokenized["question_tokens"]
    question_start_of_word = sample.tokenized["question_start_of_word"]
    question_len_t = len(question_tokens)
    passage_start_t = sample.tokenized["passage_start_t"]
    passage_tokens = sample.tokenized["passage_tokens"]
    passage_start_of_word = sample.tokenized["passage_start_of_word"]
    passage_len_t = len(passage_tokens)
    answers = sample.tokenized["answers"]

    # Turn sample_id into a list of ints (len 3) where the ints are the two halves of the squad_id
    # converted to base 10 (see utils.encode_squad_id) and the passage_id
    sample_id = [int(x) for x in sample.id.split("-")]

    # Generates a numpy array of shape (max_answers, 2) where (i, 2) indexes into the start and end indixes
    # of the ith answer. The array is filled with -1 since the number of answers is often less than max_answers
    # no answer labels are represented by (0,0)
    labels = generate_labels(answers,
                             passage_len_t,
                             question_len_t,
                             tokenizer,
                             max_answers)

    # Generate a start of word vector for the full sequence (i.e. question + answer + special tokens).
    # This will allow us to perform evaluation during training without clear text.
    # Note that in the current implementation, special tokens do not count as start of word.
    start_of_word = combine_vecs(question_start_of_word, passage_start_of_word, tokenizer, spec_tok_val=0)

    # Combines question_tokens and passage_tokens (str) into a single encoded vector of token indices (int)
    # called input_ids. This encoded vector also contains special tokens (e.g. [CLS]). It will have length =
    # (question_len_t + passage_len_t + n_special_tokens). This may be less than max_seq_len but will not be greater
    # than max_seq_len since truncation was already performed when the document was chunked into passages
    # (c.f. create_samples_squad() )
    encoded = tokenizer.encode_plus(text=sample.tokenized["question_tokens"],
                                    text_pair=sample.tokenized["passage_tokens"],
                                    add_special_tokens=True,
                                    max_length=None,
                                    truncation_strategy='only_second',
                                    return_tensors=None)
    input_ids = encoded["input_ids"]
    segment_ids = encoded["token_type_ids"]

    # seq_2_start_t is the index of the first token in the second text sequence (e.g. passage)
    if tokenizer.__class__.__name__ == "RobertaTokenizer":
        seq_2_start_t = get_roberta_seq_2_start(input_ids)
    else:
        seq_2_start_t = segment_ids.index(1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
    pad_idx = tokenizer.pad_token_id
    padding = [pad_idx] * (max_seq_len - len(input_ids))
    zero_padding = [0] * (max_seq_len - len(input_ids))

    input_ids += padding
    padding_mask += zero_padding
    segment_ids += zero_padding
    start_of_word += zero_padding

    # The XLM-Roberta tokenizer generates a segment_ids vector that separates the first sequence from the second.
    # However, when this is passed in to the forward fn of the Roberta model, it throws an error since
    # Roberta has only a single token embedding (!!!). To get around this, we want to have a segment_ids
    # vec that is only 0s
    if tokenizer.__class__.__name__ == "XLMRobertaTokenizer":
        segment_ids = np.zeros_like(segment_ids)

    # Todo: explain how only the first of labels will be used in train, and the full array will be used in eval
    # TODO Offset, start of word and spec_tok_mask are not actually needed by model.forward() but are needed for model.formatted_preds()
    # TODO passage_start_t is index of passage's first token  relative to document
    # I don't think we actually need offsets anymore
    feature_dict = {"input_ids": input_ids,
                    "padding_mask": padding_mask,
                    "segment_ids": segment_ids,
                    "is_impossible": is_impossible,
                    "id": sample_id,
                    "passage_start_t": passage_start_t,
                    "start_of_word": start_of_word,
                    "labels": labels,
                    "seq_2_start_t": seq_2_start_t}
    return [feature_dict]


def generate_labels(answers, passage_len_t, question_len_t, tokenizer, max_answers):
    """
    Creates QA label for each answer in answers. The labels are the index of the start and end token
    relative to the passage. They are contained in an array of size (max_answers, 2).
    -1 used to fill array since there the number of answers is often less than max_answers.
    The index values take in to consideration the question tokens, and also special tokens such as [CLS].
    When the answer is not fully contained in the passage, or the question
    is impossible to answer, the start_idx and end_idx are 0 i.e. start and end are on the very first token
    (in most models, this is the [CLS] token). """

    label_idxs = np.full((max_answers, 2), fill_value=-1)

    # If there are no answers
    if len(answers) == 0:
        label_idxs[0, :] = 0
        return label_idxs

    for i, answer in enumerate(answers):
        start_idx = answer["start_t"]
        end_idx = answer["end_t"]

        # We are going to operate on one-hot label vectors which will later be converted back to label indices.
        # This is to take advantage of tokenizer.encode_plus() which applies model dependent special token conventions.
        # The two label vectors (start and end) are composed of sections that correspond to the question and
        # passage tokens. These are initialized here. The section corresponding to the question
        # will always be composed of 0s.
        start_vec_question = [0] * question_len_t
        end_vec_question = [0] * question_len_t
        start_vec_passage = [0] * passage_len_t
        end_vec_passage = [0] * passage_len_t

        # If the answer is in the current passage, populate the label vector with 1s for start and end
        if answer_in_passage(start_idx, end_idx, passage_len_t):
            start_vec_passage[start_idx] = 1
            end_vec_passage[end_idx] = 1

        # Combine the sections of the label vectors. The length of each of these will be:
        # question_len_t + passage_len_t + n_special_tokens
        start_vec = combine_vecs(start_vec_question,
                                    start_vec_passage,
                                    tokenizer,
                                    spec_tok_val=0)
        end_vec = combine_vecs(end_vec_question,
                                  end_vec_passage,
                                  tokenizer,
                                  spec_tok_val=0)

        start_label_present = 1 in start_vec
        end_label_present = 1 in end_vec
        # This is triggered if the answer is not in the passage or the question is_impossible
        # In both cases, the token at idx=0 (in BERT, this is the [CLS] token) is given both the start and end label
        if start_label_present is False and end_label_present is False:
            start_vec[0] = 1
            end_vec[0] = 1
        elif start_label_present is False or end_label_present is False:
            raise Exception("The label vectors are lacking either a start or end label")

        # Ensure label vectors are one-hot
        assert sum(start_vec) == 1
        assert sum(end_vec) == 1


        start_idx = start_vec.index(1)
        end_idx = end_vec.index(1)

        label_idxs[i, 0] = start_idx
        label_idxs[i, 1] = end_idx

    assert np.max(label_idxs) > -1

    return label_idxs


def combine_vecs(question_vec, passage_vec, tokenizer, spec_tok_val=-1):
    """ Combine a question_vec and passage_vec in a style that is appropriate to the model. Will add slots in
    the returned vector for special tokens like [CLS] where the value is determine by spec_tok_val."""

    # Join question_label_vec and passage_label_vec and add slots for special tokens
    vec = tokenizer.build_inputs_with_special_tokens(token_ids_0=question_vec,
                                                     token_ids_1=passage_vec)
    spec_toks_mask = tokenizer.get_special_tokens_mask(token_ids_0=question_vec,
                                                       token_ids_1=passage_vec)

    # If a value in vec corresponds to a special token, it will be replaced with spec_tok_val
    combined = [v if not special_token else spec_tok_val for v, special_token in zip(vec, spec_toks_mask)]

    return combined


def answer_in_passage(start_idx, end_idx, passage_len):
    if passage_len > start_idx > 0 and passage_len > end_idx > 0:
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

def sample_to_features_squadOLD(
    sample, tokenizer, max_seq_len, doc_stride, max_query_length, tasks,
):
    sample.clear_text = DotMap(sample.clear_text, _dynamic=False)
    is_training = sample.clear_text.is_training

    unique_id = 1000000000
    features = []

    query_tokens = tokenizer.tokenize(sample.clear_text.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(sample.clear_text.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and sample.clear_text.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not sample.clear_text.is_impossible:
        tok_start_position = orig_to_tok_index[sample.clear_text.start_position]
        if sample.clear_text.end_position < len(sample.clear_text.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[sample.clear_text.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _SQUAD_improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            tokenizer,
            sample.clear_text.orig_answer_text,
        )

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_len - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"]
    )
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            padding_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(padding_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        start_position = 0
        end_position = 0
        if is_training and not sample.clear_text.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we keep it but set the start and end position to unanswerable
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if is_training and sample.clear_text.is_impossible:
            start_position = 0
            end_position = 0

        inp_feat = {}
        inp_feat["input_ids"] = input_ids
        inp_feat["padding_mask"] = padding_mask  # attention_mask
        inp_feat["segment_ids"] = segment_ids  # token_type_ids
        inp_feat["start_position"] = start_position
        inp_feat["end_position"] = end_position
        inp_feat["is_impossible"] = sample.clear_text.is_impossible
        inp_feat["sample_id"] = sample.id
        inp_feat["passage_shift"] = doc_span.start
        features.append(inp_feat)
        unique_id += 1

    return features


def _SQUAD_improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)
