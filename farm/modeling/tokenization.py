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

import collections
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from transformers.tokenization_albert import AlbertTokenizer
from transformers.tokenization_bert import BertTokenizer, BertTokenizerFast, load_vocab
from transformers.tokenization_distilbert import DistilBertTokenizer, DistilBertTokenizerFast
from transformers.tokenization_electra import ElectraTokenizer, ElectraTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.tokenization_camembert import CamembertTokenizer

from farm.modeling.wordembedding_utils import load_from_cache, EMBEDDING_VOCAB_FILES_MAP, run_split_on_punc


logger = logging.getLogger(__name__)

# Special characters used by the different tokenizers to indicate start of word / whitespace
SPECIAL_TOKENIZER_CHARS = r"^(##|Ġ|▁)"


class Tokenizer:
    """
    Simple Wrapper for Tokenizers from the transformers package. Enables loading of different Tokenizer classes with a uniform interface.
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path, tokenizer_class=None, use_fast=False, **kwargs):
        """
        Enables loading of different Tokenizer classes with a uniform interface. Either infer the class from
        `pretrained_model_name_or_path` or define it manually via `tokenizer_class`.

        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (e.g. `bert-base-uncased`)
        :type pretrained_model_name_or_path: str
        :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
        :type tokenizer_class: str
        :param use_fast: (Optional, False by default) Indicate if FARM should try to load the fast version of the tokenizer (True) or
            use the Python one (False).
            Only DistilBERT, BERT and Electra fast tokenizers are supported.
        :type use_fast: bool
        :param kwargs:
        :return: Tokenizer
        """

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        # guess tokenizer type from name
        if tokenizer_class is None:
            if "albert" in pretrained_model_name_or_path.lower():
                tokenizer_class = "AlbertTokenizer"
            elif "xlm-roberta" in pretrained_model_name_or_path.lower():
                tokenizer_class = "XLMRobertaTokenizer"
            elif "roberta" in pretrained_model_name_or_path.lower():
                tokenizer_class = "RobertaTokenizer"
            elif 'codebert' in pretrained_model_name_or_path.lower():
                    if "mlm" in pretrained_model_name_or_path.lower():
                        raise NotImplementedError("MLM part of codebert is currently not supported in FARM")
                    else:
                        tokenizer_class = "RobertaTokenizer"
            elif "camembert" in pretrained_model_name_or_path.lower() or "umberto" in pretrained_model_name_or_path:
                tokenizer_class = "CamembertTokenizer"
            elif "distilbert" in pretrained_model_name_or_path.lower():
                tokenizer_class = "DistilBertTokenizer"
            elif "bert" in pretrained_model_name_or_path.lower():
                tokenizer_class = "BertTokenizer"
            elif "xlnet" in pretrained_model_name_or_path.lower():
                tokenizer_class = "XLNetTokenizer"
            elif "electra" in pretrained_model_name_or_path.lower():
                tokenizer_class = "ElectraTokenizer"
            elif "word2vec" in pretrained_model_name_or_path.lower() or \
                    "glove" in pretrained_model_name_or_path.lower() or \
                    "fasttext" in pretrained_model_name_or_path.lower():
                tokenizer_class = "EmbeddingTokenizer"
            elif "minilm" in pretrained_model_name_or_path.lower():
                tokenizer_class = "BertTokenizer"
            else:
                raise ValueError(f"Could not infer tokenizer_class from name '{pretrained_model_name_or_path}'. Set "
                                 f"arg `tokenizer_class` in Tokenizer.load() to one of: AlbertTokenizer, "
                                 f"XLMRobertaTokenizer, RobertaTokenizer, DistilBertTokenizer, BertTokenizer, or "
                                 f"XLNetTokenizer.")
            logger.info(f"Loading tokenizer of type '{tokenizer_class}'")
        # return appropriate tokenizer object
        ret = None
        if tokenizer_class == "AlbertTokenizer":
            if use_fast:
                logger.error('AlbertTokenizerFast is not supported! Using AlbertTokenizer instead.')
                ret = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
            else:
                ret = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True,  **kwargs)
        elif tokenizer_class == "XLMRobertaTokenizer":
            if use_fast:
                logger.error('XLMRobertaTokenizerFast is not supported! Using XLMRobertaTokenizer instead.')
                ret = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "RobertaTokenizer" in tokenizer_class:  # because it also might be fast tokekenizer we use "in"
            if use_fast:
                logger.error('RobertaTokenizerFast is not supported! Using RobertaTokenizer instead.')
                ret = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "DistilBertTokenizer" in tokenizer_class:  # because it also might be fast tokekenizer we use "in"
            if use_fast:
                ret = DistilBertTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "BertTokenizer" in tokenizer_class:  # because it also might be fast tokekenizer we use "in"
            if use_fast:
                ret = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == "XLNetTokenizer":
            if use_fast:
                logger.error('XLNetTokenizerFast is not supported! Using XLNetTokenizer instead.')
                ret = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
            else:
                ret = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
        elif "ElectraTokenizer" in tokenizer_class:  # because it also might be fast tokekenizer we use "in"
            if use_fast:
                ret = ElectraTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == "EmbeddingTokenizer":
            if use_fast:
                logger.error('EmbeddingTokenizerFast is not supported! Using EmbeddingTokenizer instead.')
                ret = EmbeddingTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = EmbeddingTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == "CamembertTokenizer":
            if use_fast:
                logger.error('CamembertTokenizerFast is not supported! Using CamembertTokenizer instead.')
                ret = CamembertTokenizer._from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = CamembertTokenizer._from_pretrained(pretrained_model_name_or_path, **kwargs)
        if ret is None:
            raise Exception("Unable to load tokenizer")
        else:
            return ret


class EmbeddingTokenizer(PreTrainedTokenizer):
    """Constructs an EmbeddingTokenizer.
    """

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        """
        :param vocab_file: Path to a one-word-per-line vocabulary file
        :type vocab_file: str
        :param do_lower_case: Flag whether to lower case the input
        :type do_lower_case: bool
        """
        # TODO check why EmbeddingTokenizer.tokenize gives many UNK, while tokenize_with_metadata() works fine

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.unk_tok_idx = self.vocab[unk_token]
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_lower_case = do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the tokenizer from local path or remote."""
        if pretrained_model_name_or_path in EMBEDDING_VOCAB_FILES_MAP["vocab_file"]:
            # Get the vocabulary from AWS S3 bucket or cache
            resolved_vocab_file = load_from_cache(pretrained_model_name_or_path,
                                                  EMBEDDING_VOCAB_FILES_MAP["vocab_file"],
                                                  **kwargs)
        elif os.path.isdir(pretrained_model_name_or_path):
            # Get the vocabulary from local files
            logger.info(
                f"Model name '{pretrained_model_name_or_path}' not found in model shortcut name "
                f"list ({', '.join(EMBEDDING_VOCAB_FILES_MAP['vocab_file'].keys())}). "
                f"Assuming '{pretrained_model_name_or_path}' is a path to a directory containing tokenizer files.")

            temp = open(str(Path(pretrained_model_name_or_path) / "language_model_config.json"), "r",
                        encoding="utf-8").read()
            config_dict = json.loads(temp)

            resolved_vocab_file = str(Path(pretrained_model_name_or_path) / config_dict["vocab_filename"])
        else:
            logger.error(
                f"Model name '{pretrained_model_name_or_path}' not found in model shortcut name "
                f"list ({', '.join(EMBEDDING_VOCAB_FILES_MAP['vocab_file'].keys())}) nor as local folder ")
            raise NotImplementedError

        tokenizer = cls(vocab_file=resolved_vocab_file, **kwargs)
        return tokenizer

    def _tokenize(self, text, **kwargs):
        if self.do_lower_case:
            text = text.lower()
        tokens = run_split_on_punc(text)
        tokens = [t if t in self.vocab else self.unk_token for t in tokens]
        return tokens

    def save_pretrained(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_tok_idx)


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
    # Fast Tokenizers return offsets, so we don't need to calculate them ourselves
    if tokenizer.is_fast:
        tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        tokens = []
        offsets = []
        start_of_word = []
        previous_token_end = -1
        for token_id, is_special_token, offset in zip(tokenized["input_ids"],
                                                      tokenized["special_tokens_mask"],
                                                      tokenized["offset_mapping"]):
            if is_special_token == 0:
                tokens.append(tokenizer.decode([token_id]))
                offsets.append(offset[0])
                start_of_word.append(True if offset[0] != previous_token_end else False)
                previous_token_end = offset[1]
        tokenized = {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}
    else:
        # normalize all other whitespace characters to " "
        # Note: using text.split() directly would destroy the offset,
        # since \n\n\n would be treated similarly as a single \n
        text = re.sub(r"\s", " ", text)
        # split text into "words" (here: simple whitespace tokenizer).
        words = text.split(" ")
        word_offsets = []
        cumulated = 0
        for idx, word in enumerate(words):
            word_offsets.append(cumulated)
            cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

        # split "words" into "subword tokens"
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
    idx = 0
    for w, w_off in zip(words, word_offsets):
        idx += 1
        if idx % 500000 == 0:
            logger.info(idx)
        # Get (subword) tokens of single word.

        # empty / pure whitespace
        if len(w) == 0:
          continue
        # For the first word of a text: we just call the regular tokenize function.
        # For later words: we need to call it with add_prefix_space=True to get the same results with roberta / gpt2 tokenizer
        # see discussion here. https://github.com/huggingface/transformers/issues/1196
        elif len(tokens) == 0:
            tokens_word = tokenizer.tokenize(w)
        else:
            if type(tokenizer) == RobertaTokenizer:
                tokens_word = tokenizer.tokenize(w, add_prefix_space=True)
            else:
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
            # Don't use length of unk token for offset calculation
            if orig_tok == tokenizer.special_tokens_map["unk_token"]:
                w_off += 1
            else:
                w_off += len(orig_tok)
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)

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
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=pair) if with_special_tokens else 0
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
