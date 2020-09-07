"""
Contains functions that turn readable clear text input into dictionaries of features
"""


import logging
import re
import collections
from dotmap import DotMap
import numpy as np

from farm.data_handler.samples import Sample
from farm.data_handler.utils import (
    expand_labels,
    pad,
    mask_random_words)
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

    if tokenizer.is_fast:
        text = sample.clear_text["text"]
        # Here, we tokenize the sample for the second time to get all relevant ids
        # This should change once we git rid of FARM's tokenize_with_metadata()
        inputs = tokenizer(text,
                           return_token_type_ids=True,
                           truncation=True,
                           truncation_strategy="longest_first",
                           max_length=max_seq_len,
                           return_special_tokens_mask=True)

        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != len(sample.tokenized["tokens"]):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to "
                         f"{len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs "
                         f"from number of tokens produced in tokenize_with_metadata(). \n"
                         f"Further processing is likely to be wrong.")
    else:
        # TODO It might be cleaner to adjust the data structure in sample.tokenized
        tokens_a = sample.tokenized["tokens"]
        tokens_b = sample.tokenized.get("tokens_b", None)

        inputs = tokenizer.encode_plus(
            tokens_a,
            tokens_b,
            add_special_tokens=True,
            truncation=False,  # truncation_strategy is deprecated
            return_token_type_ids=True,
            is_pretokenized=False,
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

    if tokenizer.is_fast:
        text = sample.clear_text["text"]
        # Here, we tokenize the sample for the second time to get all relevant ids
        # This should change once we git rid of FARM's tokenize_with_metadata()
        inputs = tokenizer(text,
                           return_token_type_ids=True,
                           truncation=True,
                           truncation_strategy="longest_first",
                           max_length=max_seq_len,
                           return_special_tokens_mask=True)

        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != len(sample.tokenized["tokens"]):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to "
                         f"{len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs "
                         f"from number of tokens produced in tokenize_with_metadata().\n"
                         f"Further processing is likely to be wrong!")
    else:
        inputs = tokenizer.encode_plus(text=tokens,
                                       text_pair=None,
                                       add_special_tokens=True,
                                       truncation=False,
                                       return_special_tokens_mask=True,
                                       return_token_type_ids=True,
                                       is_pretokenized=False
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

        if tokenizer.is_fast:
            # Detokenize input as fast tokenizer can't handle tokenized input
            tokens_a = " ".join(tokens_a)
            tokens_a = re.sub(r"(^|\s)(##)", "", tokens_a)
            tokens_b = " ".join(tokens_b)
            tokens_b = re.sub(r"(^|\s)(##)", "", tokens_b)

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
        if tokenizer.is_fast:
            # Detokenize input as fast tokenizer can't handle tokenized input
            tokens_a = " ".join(tokens_a)
            tokens_a = re.sub(r"(^|\s)(##)", "", tokens_a)

        # convert lm labels to ids
        lm_label_ids = [-1 if tok == '' else tokenizer.convert_tokens_to_ids(tok) for tok in t1_label]

    if tokenizer.is_fast:
        inputs = tokenizer(text=tokens_a,
                           text_pair=tokens_b,
                           add_special_tokens=True,
                           return_special_tokens_mask=True,
                           return_token_type_ids=True)

        seq_b_len = len(sample.tokenized["text_b"]["tokens"]) if "text_b" in sample.tokenized else 0
        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != \
           (len(sample.tokenized["text_a"]["tokens"]) + seq_b_len):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to "
                         f"{len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs "
                         f"from number of tokens produced in tokenize_with_metadata(). \n"
                         f"Further processing is likely to be wrong.")
    else:
        # encode string tokens to input_ids and add special tokens
        inputs = tokenizer.encode_plus(text=tokens_a,
                                       text_pair=tokens_b,
                                       add_special_tokens=True,
                                       truncation=False,
                                       truncation_strategy='do_not_truncate',
                                       # We've already truncated our tokens before
                                       return_special_tokens_mask=True,
                                       return_token_type_ids=True
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


def sample_to_features_qa(sample, tokenizer, max_seq_len, sp_toks_start, sp_toks_mid,
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
    if tokenizer.__class__.__name__ in ["XLMRobertaTokenizer", "RobertaTokenizer"]:
        segment_ids = np.zeros_like(segment_ids)

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
                    "seq_2_start_t": seq_2_start_t}
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
