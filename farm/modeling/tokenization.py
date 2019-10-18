# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
import unicodedata
import re
import numpy as np

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer

logger = logging.getLogger(__name__)


# Simple wrapper for transformers tokenizer to simplify loading of model specific tokenizers
class Tokenizer():
    @classmethod
    def load(cls, pretrained_model_name_or_path, tokenizer_class=None, **kwargs):
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
            orig_tok = re.sub(r"^(##|Ġ|▁)", "", tok)
            w_off += len(orig_tok)
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)

    assert len(tokens) == len(token_offsets) == len(start_of_word)
    return tokens, token_offsets, start_of_word


def truncate_sequences(seq_a, seq_b, tokenizer, max_seq_len, truncation_strategy='longest_first', with_special_tokens=True, stride=0):
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
    new_seq = seq.copy()
    special_tokens_indices = np.where(np.array(special_tokens_mask) == 1)[0]
    for idx in special_tokens_indices:
        new_seq.insert(idx, insert_element)
    return new_seq