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

from pytorch_transformers.tokenization_bert import BertTokenizer, WordpieceTokenizer

logger = logging.getLogger(__name__)


class BertTokenizer(BertTokenizer):
    def add_custom_vocab(self, custom_vocab_file):
        self.vocab = self._load_custom_vocab(custom_vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def _load_custom_vocab(self, custom_vocab_file):
        custom_vocab = {}
        unique_custom_tokens = set()
        idx = 0
        with open(custom_vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break

                if token not in unique_custom_tokens:
                    if token not in self.vocab.keys():
                        key = "[unused{}]".format(idx)
                        custom_vocab[key] = token.strip()
                        idx += 1
                        unique_custom_tokens.add(token)
                    else:
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
