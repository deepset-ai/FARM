# coding=utf-8
# Copyright 2018 deepset team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import re
import numpy as np

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer

logger = logging.getLogger(__name__)

# Special characters used by the different tokenizers to indicate start of word / whitespace
SPECIAL_TOKENIZER_CHARS = r"^(##|Ġ|▁)"


class Tokenizer:
    """
    Simple Wrapper for Tokenizers from the transformers package. Enables loading of different Tokenizer classes with a uniform interface.
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path, tokenizer_class=None, **kwargs):
        """
        Enables loading of different Tokenizer classes with a uniform interface. Either infer the class from
        `pretrained_model_name_or_path` or define it manually via `tokenizer_class`.

        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (e.g. `bert-base-uncased`)
        :type pretrained_model_name_or_path: str
        :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
        :type tokenizer_class: str
        :param kwargs:
        :return: Tokenizer
        """
        # guess tokenizer type from name
        if tokenizer_class is None:
            if "roberta" in pretrained_model_name_or_path.lower():
                tokenizer_class = "RobertaTokenizer"
            elif "bert" in pretrained_model_name_or_path.lower():
                tokenizer_class = "BertTokenizer"
            elif "xlnet" in pretrained_model_name_or_path.lower():
                tokenizer_class = "XLNetTokenizer"
            else:
                raise ValueError(f"Could not infer tokenizer_type from name '{pretrained_model_name_or_path}'. Set arg `tokenizer_type` in Tokenizer.load() to one of: 'bert', 'roberta', 'xlnet' ")
            logger.info(f"Loading tokenizer of type '{tokenizer_class}'")
        # return appropriate tokenizer object
        if tokenizer_class == "RobertaTokenizer":
            return RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == "BertTokenizer":
            return BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == "XLNetTokenizer":
            return XLNetTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


def tokenize_with_metadata(text, tokenizer):
    """
    Performing tokenization while storing some important metadata for each token:

    * offsets: (int) Character index where the token begins in the original text
    * start_of_word: (bool) If the token is the start of a word. Particularly helpful for NER and QA tasks.

    We do this by first doing whitespace tokenization and then applying the model specific tokenizer to each "word".

    .. note::  We don't assume to preserve exact whitespaces in the tokens!
               This means: tabs, new lines, multiple whitespace etc will all resolve to a single " ".
               This doesn't make a difference for BERT + XLNet but it does for RoBERTa.
               For RoBERTa it has the positive effect of a shorter sequence length, but some information about whitespace
               type is lost which might be helpful for certain NLP tasks ( e.g tab for tables).

    :param text: Text to tokenize
    :type text: str
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :return: Dictionary with "tokens", "offsets" and "start_of_word"
    :rtype: dict

    """

    # split text into "words" (here: simple whitespace tokenizer)
    words = text.split(" ")
    word_offsets = []
    cumulated = 0
    for idx, word in enumerate(words):
        word_offsets.append(cumulated)
        cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

    # split "words"into "subword tokens"
    tokens, offsets, start_of_word = _words_to_tokens(
        words, word_offsets, tokenizer
    )

    tokenized = {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}
    return tokenized


def _words_to_tokens(words, word_offsets, tokenizer):
    """
    Tokenize "words" into subword tokens while keeping track of offsets and if a token is the start of a word.

    :param words: list of words.
    :type words: list
    :param word_offsets: Character indices where each word begins in the original text
    :type word_offsets: list
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :return: tokens, offsets, start_of_word

    """
    tokens = []
    token_offsets = []
    start_of_word = []
    for w, w_off in zip(words, word_offsets):
        # Get (subword) tokens of single word.
        # For the first word of a text: we just call the regular tokenize function.
        # For later words: we need to call it with add_prefix_space=True to get the same results with roberta / gpt2 tokenizer
        # see discussion here. https://github.com/huggingface/transformers/issues/1196
        if len(tokens) == 0:
            tokens_word = tokenizer.tokenize(w)
        else:
            try:
                tokens_word = tokenizer.tokenize(w, add_prefix_space=True)
            except TypeError:
                tokens_word = tokenizer.tokenize(w)

        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word

        # get global offset for each token in word + save marker for first tokens of a word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            # Depending on the tokenizer type special chars are added to distinguish tokens with preceeding
            # whitespace (=> "start of a word"). We need to get rid of these to calculate the original length of the token
            orig_tok = re.sub(SPECIAL_TOKENIZER_CHARS, "", tok)
            w_off += len(orig_tok)
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)

    assert len(tokens) == len(token_offsets) == len(start_of_word)
    return tokens, token_offsets, start_of_word


def truncate_sequences(seq_a, seq_b, tokenizer, max_seq_len, truncation_strategy='longest_first',
                       with_special_tokens=True, stride=0):
    """
    Reduces a single sequence or a pair of sequences to a maximum sequence length.
    The sequences can contain tokens or any other elements (offsets, masks ...).
    If `with_special_tokens` is enabled, it'll remove some additional tokens to have exactly enough space for later adding special tokens (CLS, SEP etc.)

    Supported truncation strategies:

    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one at each token (when there is a pair of input sequences). Overflowing tokens only contains overflow from the first sequence.
    - only_first: Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
    - only_second: Only truncate the second sequence
    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)

    :param seq_a: First sequence of tokens/offsets/...
    :type seq_a: list
    :param seq_b: Optional second sequence of tokens/offsets/...
    :type seq_b: None or list
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :param max_seq_len:
    :type max_seq_len: int
    :param truncation_strategy: how the sequence(s) should be truncated down. Default: "longest_first" (see above for other options).
    :type truncation_strategy: str
    :param with_special_tokens: If true, it'll remove some additional tokens to have exactly enough space for later adding special tokens (CLS, SEP etc.)
    :type with_special_tokens: bool
    :param stride: optional stride of the window during truncation
    :type stride: int
    :return: truncated seq_a, truncated seq_b, overflowing tokens

    """
    pair = bool(seq_b is not None)
    len_a = len(seq_a)
    len_b = len(seq_b) if pair else 0
    num_special_tokens = tokenizer.num_added_tokens(pair=pair) if with_special_tokens else 0
    total_len = len_a + len_b + num_special_tokens
    overflowing_tokens = []

    if max_seq_len and total_len > max_seq_len:
        seq_a, seq_b, overflowing_tokens = tokenizer.truncate_sequences(seq_a, pair_ids=seq_b,
                                                                    num_tokens_to_remove=total_len - max_seq_len,
                                                                    truncation_strategy=truncation_strategy,
                                                                    stride=stride)
    return (seq_a, seq_b, overflowing_tokens)


def insert_at_special_tokens_pos(seq, special_tokens_mask, insert_element):
    """
    Adds elements to a sequence at the positions that align with special tokens.
    This is useful for expanding label ids or masks, so that they align with corresponding tokens (incl. the special tokens)

    Example:

    .. code-block:: python

      # Tokens:  ["CLS", "some", "words","SEP"]
      >>> special_tokens_mask =  [1,0,0,1]
      >>> lm_label_ids =  [12,200]
      >>> insert_at_special_tokens_pos(lm_label_ids, special_tokens_mask, insert_element=-1)
      [-1, 12, 200, -1]

    :param seq: List where you want to insert new elements
    :type seq: list
    :param special_tokens_mask: list with "1" for positions of special chars
    :type special_tokens_mask: list
    :param insert_element: the value you want to insert
    :return: list

    """
    new_seq = seq.copy()
    special_tokens_indices = np.where(np.array(special_tokens_mask) == 1)[0]
    for idx in special_tokens_indices:
        new_seq.insert(idx, insert_element)
    return new_seq
