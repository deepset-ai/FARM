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

import collections
import logging
from io import open
import os
import unicodedata

from transformers.tokenization_bert import BertTokenizer, WordpieceTokenizer, BasicTokenizer, load_vocab

logger = logging.getLogger(__name__)


class BasicTokenizer(BasicTokenizer):
    def __init__(self, do_lower_case=True, never_split=None, never_split_chars=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be desactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.never_split_chars = never_split_chars

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char, excluded_chars=self.never_split_chars):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


class BertTokenizer(BertTokenizer):

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, never_split_chars=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        """Constructs a BertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be desactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        super(BertTokenizer, self).__init__(vocab_file, do_lower_case=do_lower_case, do_basic_tokenize=True, never_split=never_split, never_split_chars=never_split_chars,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=tokenize_chinese_chars, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split,
                                                  never_split_chars=never_split_chars,
                                                  tokenize_chinese_chars=tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
        assert len(self.vocab) > 0
        assert self.wordpiece_tokenizer is not None


    def add_custom_vocab(self, custom_vocab_file):
        self.vocab = self._load_custom_vocab(custom_vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def _load_custom_vocab(self, custom_vocab_file):
        custom_vocab = {}
        unique_custom_tokens = set()
        idx = 1
        num_dropped = 0
        with open(custom_vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline().strip()
                if not token:
                    break

                if token not in unique_custom_tokens:
                    if token not in self.vocab.keys():
                        key = "[unused{}]".format(idx)
                        custom_vocab[key] = token
                        idx += 1
                        unique_custom_tokens.add(token)
                    else:
                        num_dropped += 1
                        logger.info("Dropped custom token (already in original vocab): {}".format(token))
                else:
                    logger.info("Dropped custom token (duplicate): {}".format(token))
        # merge vocabs
        update_count = 0
        updated_vocab = []
        for k,v in self.vocab.items():
            if k in custom_vocab.keys():
                updated_vocab.append((custom_vocab[k], v))
                update_count += 1
            else:
                updated_vocab.append((k, v))
        self.vocab = collections.OrderedDict(updated_vocab)

        if update_count < len(custom_vocab):
            logger.warning("Updated vocabulary only with {} out of {} tokens from supplied custom vocabulary. The original vocab might not have enough unused tokens.".format(update_count, len(custom_vocab)))
        else:
            logger.info("Updated vocabulary with {} out of {} tokens from custom vocabulary.".format(update_count, len(custom_vocab)))
        if(num_dropped > 0):
            logger.info(f"Dropped {num_dropped} items from custom vocab, because they were already contained in original vocab.")

        return self.vocab




def tokenize_with_metadata(text, tokenizer, max_seq_len):
    # split text into "words" (here: simple whitespace tokenizer)
    words = text.split(" ")
    word_offsets = []
    cumulated = 0
    for idx, word in enumerate(words):
        word_offsets.append(cumulated)
        cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

    # split "words"into "subword tokens"
    tokens, offsets, start_of_word = _words_to_tokens(
        words, word_offsets, tokenizer, max_seq_len
    )

    tokenized = {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}
    return tokenized


def _words_to_tokens(words, word_offsets, tokenizer, max_seq_len):
    tokens = []
    token_offsets = []
    start_of_word = []
    for w, w_off in zip(words, word_offsets):
        # Get tokens of single word
        tokens_word = tokenizer.tokenize(w)

        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word

        # get gloabl offset for each token in word + save marker for first tokens of a word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            w_off += len(tok.replace("##", ""))
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)

    # Clip at max_seq_length. The "-2" is for CLS and SEP token
    tokens = tokens[: max_seq_len - 2]
    token_offsets = token_offsets[: max_seq_len - 2]
    start_of_word = start_of_word[: max_seq_len - 2]

    assert len(tokens) == len(token_offsets) == len(start_of_word)
    return tokens, token_offsets, start_of_word

def _is_punctuation(char, excluded_chars=None):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if excluded_chars:
        if char in excluded_chars:
            return False

    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False