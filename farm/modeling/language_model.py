# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,  The HuggingFace Inc. Team and deepset Team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Acknowledgements: Many of the modeling parts here come from the great transformers repository: https://github.com/huggingface/transformers.
Thanks for the great work! """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

from transformers.modeling_bert import BertModel, BertConfig
from transformers.modeling_roberta import RobertaModel, RobertaConfig
from transformers.modeling_xlnet import XLNetModel, XLNetConfig
from transformers.modeling_albert import AlbertModel, AlbertConfig
from transformers.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from transformers.modeling_distilbert import DistilBertModel, DistilBertConfig
from transformers.modeling_utils import SequenceSummary

# These are the names of the attributes in various model configs which refer to the number of dimensions
# in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]

class LanguageModel(nn.Module):
    """
    The parent class for any kind of model that can embed language into a semantic vector space. Practically
    speaking, these models read in tokenized sentences and return vectors that capture the meaning of sentences
    or of tokens.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def forward(self, input_ids, padding_mask, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_scratch(cls, model_type, vocab_size):
        if model_type.lower() == "bert":
            model = Bert
        return model.from_scratch(vocab_size)

    @classmethod
    def load(cls, pretrained_model_name_or_path, n_added_tokens=0, **kwargs):
        """
        Load a pretrained language model either by

        1. specifying its name and downloading it
        2. or pointing to the directory it is saved in.

        Available remote models:

        * bert-base-uncased
        * bert-large-uncased
        * bert-base-cased
        * bert-large-cased
        * bert-base-multilingual-uncased
        * bert-base-multilingual-cased
        * bert-base-chinese
        * bert-base-german-cased
        * roberta-base
        * roberta-large
        * xlnet-base-cased
        * xlnet-large-cased
        * xlm-roberta-base
        * xlm-roberta-large
        * albert-base-v2
        * albert-large-v2
        * distilbert-base-german-cased
        * distilbert-base-multilingual-cased

        See all supported model variations here: https://huggingface.co/models

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(config_file):
            # it's a local directory in FARM format
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path)
        else:
            # it's transformers format (either from model hub or local)
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if "xlm" in pretrained_model_name_or_path and "roberta" in pretrained_model_name_or_path:
                # TODO: for some reason, the pretrained XLMRoberta has different vocab size in the tokenizer compared to the model this is a hack to resolve that
                n_added_tokens = 3
                language_model = cls.subclasses["XLMRoberta"].load(pretrained_model_name_or_path, **kwargs)
            elif 'roberta' in pretrained_model_name_or_path:
                language_model = cls.subclasses["Roberta"].load(pretrained_model_name_or_path, **kwargs)
            elif 'albert' in pretrained_model_name_or_path:
                language_model = cls.subclasses["Albert"].load(pretrained_model_name_or_path, **kwargs)
            elif 'distilbert' in pretrained_model_name_or_path:
                language_model = cls.subclasses["DistilBert"].load(pretrained_model_name_or_path, **kwargs)
            elif 'bert' in pretrained_model_name_or_path:
                language_model = cls.subclasses["Bert"].load(pretrained_model_name_or_path, **kwargs)
            elif 'xlnet' in pretrained_model_name_or_path:
                language_model = cls.subclasses["XLNet"].load(pretrained_model_name_or_path, **kwargs)
            else:
                language_model = None

        if not language_model:
            raise Exception(
                f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved model "
                f"or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. Here's the list of available "
                f"models: https://farm.deepset.ai/api/modeling.html#farm.modeling.language_model.LanguageModel.load"
            )

        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope for custom vocab.")
            language_model.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

        return language_model

    def get_output_dims(self):
        config = self.model.config
        for odn in OUTPUT_DIM_NAMES:
            if odn in dir(config):
                return getattr(config, odn)
        else:
            raise Exception("Could not infer the output dimensions of the language model")

    def freeze(self, layers):
        """ To be implemented"""
        raise NotImplementedError()

    def unfreeze(self):
        """ To be implemented"""
        raise NotImplementedError()

    def save_config(self, save_dir):
        save_filename = Path(save_dir) / "language_model_config.json"
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        # Save Weights
        save_name = Path(save_dir) / "language_model.bin"
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        torch.save(model_to_save.state_dict(), save_name)
        self.save_config(save_dir)

    @classmethod
    def _infer_language_from_name(cls, name):
        known_languages = (
            "german",
            "english",
            "chinese",
            "indian",
            "french",
            "polish",
            "spanish",
            "multilingual",
        )
        matches = [lang for lang in known_languages if lang in name]
        if len(matches) == 0:
            language = "english"
            logger.warning(
                "Could not automatically detect from language model name what language it is. \n"
                "\t We guess it's an *ENGLISH* model ... \n"
                "\t If not: Init the language model by supplying the 'language' param."
            )
        elif len(matches) > 1:
            raise ValueError(
                "Could not automatically detect from language model name what language it is.\n"
                f"\t Found multiple matches: {matches}\n"
                "\t Please init the language model by manually supplying the 'language' as a parameter.\n"
            )
        else:
            language = matches[0]
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )

        return language

    def formatted_preds(self, input_ids, samples, extraction_strategy="pooled", extraction_layer=-1, ignore_first_token=True,
                        padding_mask=None, **kwargs):
        # get language model output from last layer
        if extraction_layer == -1:
            sequence_output, pooled_output = self.forward(input_ids, padding_mask=padding_mask, **kwargs)
        # or from earlier layer
        else:
            self.enable_hidden_states_output()
            sequence_output, pooled_output, all_hidden_states = self.forward(input_ids, padding_mask=padding_mask, **kwargs)
            sequence_output = all_hidden_states[extraction_layer]
            self.disable_hidden_states_output()
        # aggregate vectors
        if extraction_strategy == "pooled":
            if extraction_layer != -1:
                raise ValueError(f"Pooled output only works for the last layer, but got extraction_layer = {extraction_layer}. Please set `extraction_layer=-1`.)")
            vecs = pooled_output.cpu().numpy()
        elif extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()
        elif extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(sequence_output, padding_mask, extraction_strategy, ignore_first_token=ignore_first_token)
        elif extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(sequence_output, padding_mask, extraction_strategy, ignore_first_token=ignore_first_token)
        elif extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        else:
            raise NotImplementedError

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.tokenized["tokens"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output, padding_mask, strategy, ignore_first_token):

        token_vecs = sequence_output.cpu().numpy()
        # we only take the aggregated value of non-padding tokens
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        # sometimes we want to exclude the CLS token as well from our aggregation operation
        if ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]
        if strategy == "reduce_max":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
        if strategy == "reduce_mean":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data
        return pooled_vecs


class Bert(LanguageModel):
    """
    A BERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805

    """

    def __init__(self):
        super(Bert, self).__init__()
        self.model = None
        self.name = "bert"

    @classmethod
    def from_scratch(cls, vocab_size, name="bert", language="en"):
        bert = cls()
        bert.name = name
        bert.language = language
        config = BertConfig(vocab_size=vocab_size)
        bert.model = BertModel(config)
        return bert

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("bert-base-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        bert = cls()
        if "farm_lm_name" in kwargs:
            bert.name = kwargs["farm_lm_name"]
        else:
            bert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            bert_config = BertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            bert.model = BertModel.from_pretrained(farm_lm_model, config=bert_config, **kwargs)
            bert.language = bert.model.config.language
        else:
            # Pytorch-transformer Style
            bert.model = BertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            bert.language = cls._infer_language_from_name(pretrained_model_name_or_path)
        return bert

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the BERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


class Albert(LanguageModel):
    """
    An ALBERT model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    """

    def __init__(self):
        super(Albert, self).__init__()
        self.model = None
        self.name = "albert"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("albert-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        albert = cls()
        if "farm_lm_name" in kwargs:
            albert.name = kwargs["farm_lm_name"]
        else:
            albert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = AlbertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            albert.model = AlbertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            albert.language = albert.model.config.language
        else:
            # Huggingface transformer Style
            albert.model = AlbertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            albert.language = cls._infer_language_from_name(pretrained_model_name_or_path)
        return albert

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the Albert model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


class Roberta(LanguageModel):
    """
    A roberta model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692

    """

    def __init__(self):
        super(Roberta, self).__init__()
        self.model = None
        self.name = "roberta"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        roberta = cls()
        if "farm_lm_name" in kwargs:
            roberta.name = kwargs["farm_lm_name"]
        else:
            roberta.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = RobertaConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            roberta.model = RobertaModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            roberta.language = roberta.model.config.language
        else:
            # Huggingface transformer Style
            roberta.model = RobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            roberta.language = cls._infer_language_from_name(pretrained_model_name_or_path)
        return roberta

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the Roberta model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


class XLMRoberta(LanguageModel):
    """
    A roberta model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692

    """

    def __init__(self):
        super(XLMRoberta, self).__init__()
        self.model = None
        self.name = "xlm_roberta"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("xlm-roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        xlm_roberta = cls()
        if "farm_lm_name" in kwargs:
            xlm_roberta.name = kwargs["farm_lm_name"]
        else:
            xlm_roberta.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = XLMRobertaConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            xlm_roberta.model = XLMRobertaModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            xlm_roberta.language = xlm_roberta.model.config.language
        else:
            # Huggingface transformer Style
            xlm_roberta.model = XLMRobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            xlm_roberta.language = cls._infer_language_from_name(pretrained_model_name_or_path)
        return xlm_roberta

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the XLMRoberta model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


class DistilBert(LanguageModel):
    """
    A DistilBERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE: 
    - DistilBert doesn’t have token_type_ids, you don’t need to indicate which 
    token belongs to which segment. Just separate your segments with the separation 
    token tokenizer.sep_token (or [SEP])
    - Unlike the other BERT variants, DistilBert does not output the
    pooled_output. An additional pooler is initialized.
    
    """

    def __init__(self):
        super(DistilBert, self).__init__()
        self.model = None
        self.name = "distilbert"
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("distilbert-base-german-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        distilbert = cls()
        if "farm_lm_name" in kwargs:
            distilbert.name = kwargs["farm_lm_name"]
        else:
            distilbert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = AlbertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            distilbert.model = DistilBertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            distilbert.language = distilbert.model.config.language
        else:
            # Pytorch-transformer Style
            distilbert.model = DistilBertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            distilbert.language = cls._infer_language_from_name(pretrained_model_name_or_path)
        config = distilbert.model.config

        # DistilBERT does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        config.summary_last_dropout = 0
        config.summary_type = 'first'
        config.summary_activation = 'tanh'
        distilbert.pooler = SequenceSummary(config)
        distilbert.pooler.apply(distilbert.model._init_weights)
        return distilbert

    def forward(
        self,
        input_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the DistilBERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            attention_mask=padding_mask,
        )
        # We need to manually aggregate that to get a pooled output (one vec per seq)
        pooled_output = self.pooler(output_tuple[0])
        if self.model.config.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.config.output_hidden_states = False


class XLNet(LanguageModel):
    """
    A XLNet model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1906.08237
    """

    def __init__(self):
        super(XLNet, self).__init__()
        self.model = None
        self.name = "xlnet"
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("xlnet-base-cased" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        xlnet = cls()
        if "farm_lm_name" in kwargs:
            xlnet.name = kwargs["farm_lm_name"]
        else:
            xlnet.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = XLNetConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            xlnet.model = XLNetModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            xlnet.language = xlnet.model.config.language
        else:
            # Pytorch-transformer Style
            xlnet.model = XLNetModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            xlnet.language = cls._infer_language_from_name(pretrained_model_name_or_path)
            config = xlnet.model.config
        # XLNet does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the last hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        config.summary_last_dropout = 0
        xlnet.pooler = SequenceSummary(config)
        xlnet.pooler.apply(xlnet.model._init_weights)
        return xlnet

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the XLNet model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """

        # Note: XLNet has a couple of special input tensors for pretraining / text generation  (perm_mask, target_mapping ...)
        # We will need to implement them, if we wanna support LM adaptation

        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        # XLNet also only returns the sequence_output (one vec per token)
        # We need to manually aggregate that to get a pooled output (one vec per seq)
        #TODO verify that this is really doing correct pooling
        pooled_output = self.pooler(output_tuple[0])

        if self.model.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.output_hidden_states = False