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
import io
from pathlib import Path
from collections import OrderedDict

from dotmap import DotMap
from tqdm import tqdm
import copy
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

from transformers import (
    BertModel, BertConfig,
    RobertaModel, RobertaConfig,
    XLNetModel, XLNetConfig,
    AlbertModel, AlbertConfig,
    XLMRobertaModel, XLMRobertaConfig,
    DistilBertModel, DistilBertConfig,
    ElectraModel, ElectraConfig,
    CamembertModel, CamembertConfig
)

from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary
from transformers.models.bert.tokenization_bert import load_vocab
import transformers

from farm.modeling import wordembedding_utils
from farm.modeling.wordembedding_utils import s3e_pooling

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
    def load(cls, pretrained_model_name_or_path, revision=None, n_added_tokens=0, language_model_class=None, **kwargs):
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
        * google/electra-small-discriminator
        * google/electra-base-discriminator
        * google/electra-large-discriminator
        * facebook/dpr-question_encoder-single-nq-base
        * facebook/dpr-ctx_encoder-single-nq-base

        See all supported model variations here: https://huggingface.co/models

        The appropriate language model class is inferred automatically from model config
        or can be manually supplied via `language_model_class`.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param language_model_class: (Optional) Name of the language model class to load (e.g. `Bert`)
        :type language_model_class: str

        """
        kwargs["revision"] = revision
        logger.info("")
        logger.info("LOADING MODEL")
        logger.info("=============")
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(config_file):
            logger.info(f"Model found locally at {pretrained_model_name_or_path}")
            # it's a local directory in FARM format
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path)
        else:
            logger.info(f"Could not find {pretrained_model_name_or_path} locally.")
            logger.info(f"Looking on Transformers Model Hub (in local cache and online)...")
            if language_model_class is None:
                language_model_class = cls.get_language_model_class(pretrained_model_name_or_path)

            if language_model_class:
                language_model = cls.subclasses[language_model_class].load(pretrained_model_name_or_path, **kwargs)
            else:
                language_model = None

        if not language_model:
            raise Exception(
                f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved "
                f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                f"Ensure that the model class name can be inferred from the directory name when loading a "
                f"Transformers' model. Here's a list of available models: "
                f"https://farm.deepset.ai/api/modeling.html#farm.modeling.language_model.LanguageModel.load"
            )
        else:
            logger.info(f"Loaded {pretrained_model_name_or_path}")

        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab.")
            language_model.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

        return language_model

    @staticmethod
    def get_language_model_class(model_name_or_path):
        # it's transformers format (either from model hub or local)
        model_name_or_path = str(model_name_or_path)

        config = AutoConfig.from_pretrained(model_name_or_path)
        model_type = config.model_type
        if model_type == "xlm-roberta":
            language_model_class = "XLMRoberta"
        elif model_type == "roberta":
            if "mlm" in model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in FARM")
            language_model_class = "Roberta"
        elif model_type == "camembert":
            language_model_class = "Camembert"
        elif model_type == "albert":
            language_model_class = "Albert"
        elif model_type == "distilbert":
            language_model_class = "DistilBert"
        elif model_type == "bert":
            language_model_class = "Bert"
        elif model_type == "xlnet":
            language_model_class = "XLNet"
        elif model_type == "electra":
            language_model_class = "Electra"
        elif model_type == "dpr":
            if config.architectures[0] == "DPRQuestionEncoder":
                language_model_class = "DPRQuestionEncoder"
            elif config.architectures[0] == "DPRContextEncoder":
                language_model_class = "DPRContextEncoder"
            elif config.archictectures[0] == "DPRReader":
                raise NotImplementedError("DPRReader models are currently not supported.")
        else:
            # Fall back to inferring type from model name
            logger.warning("Could not infer LanguageModel class from config. Trying to infer "
                           "LanguageModel class from model name.")
            language_model_class = LanguageModel._infer_language_model_class_from_string(model_name_or_path)

        return language_model_class

    @staticmethod
    def _infer_language_model_class_from_string(model_name_or_path):
        # If inferring Language model class from config doesn't succeed,
        # fall back to inferring Language model class from model name.
        if "xlm" in model_name_or_path.lower() and "roberta" in model_name_or_path.lower():
            language_model_class = "XLMRoberta"
        elif "roberta" in model_name_or_path.lower():
            language_model_class = "Roberta"
        elif "codebert" in model_name_or_path.lower():
            if "mlm" in model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in FARM")
            else:
                language_model_class = "Roberta"
        elif "camembert" in model_name_or_path.lower() or "umberto" in model_name_or_path.lower():
            language_model_class = "Camembert"
        elif "albert" in model_name_or_path.lower():
            language_model_class = 'Albert'
        elif "distilbert" in model_name_or_path.lower():
            language_model_class = 'DistilBert'
        elif "bert" in model_name_or_path.lower():
            language_model_class = 'Bert'
        elif "xlnet" in model_name_or_path.lower():
            language_model_class = 'XLNet'
        elif "electra" in model_name_or_path.lower():
            language_model_class = 'Electra'
        elif "word2vec" in model_name_or_path.lower() or "glove" in model_name_or_path.lower():
            language_model_class = 'WordEmbedding_LM'
        elif "minilm" in model_name_or_path.lower():
            language_model_class = "Bert"
        elif "dpr-question_encoder" in model_name_or_path.lower():
            language_model_class = "DPRQuestionEncoder"
        elif "dpr-ctx_encoder" in model_name_or_path.lower():
            language_model_class = "DPRContextEncoder"
        else:
            language_model_class = None

        return language_model_class

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
    def _get_or_infer_language_from_name(cls, language, name):
        if language is not None:
            return language
        else:
            return cls._infer_language_from_name(name)

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
        if "camembert" in name:
            language = "french"
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )
        elif "umberto" in name:
            language = "italian"
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )
        elif len(matches) == 0:
            language = "english"
        elif len(matches) > 1:
            language = matches[0]
        else:
            language = matches[0]
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )

        return language

    def formatted_preds(self, logits, samples, ignore_first_token=True,
                        padding_mask=None, input_ids=None, **kwargs):
        """
        Extracting vectors from language model (e.g. for extracting sentence embeddings).
        Different pooling strategies and layers are available and will be determined from the object attributes
        `extraction_layer` and `extraction_strategy`. Both should be set via the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence
        :param samples: For each item in logits we need additional meta information to format the prediction (e.g. input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: Whether to include the first token for pooling operations (e.g. reduce_mean).
                                   Many models have here a special token like [CLS] that you don't want to include into your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. Those will also not be included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: ids of the tokens in the vocab
        :param kwargs: kwargs
        :return: list of dicts containing preds, e.g. [{"context": "some text", "vec": [-0.01, 0.5 ...]}]
        """

        if not hasattr(self, "extraction_layer") or not hasattr(self, "extraction_strategy"):
            raise ValueError("`extraction_layer` or `extraction_strategy` not specified for LM. "
                             "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`")

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ValueError(f"Pooled output only works for the last layer, but got extraction_layer = {self.extraction_layer}. Please set `extraction_layer=-1`.)")
            vecs = pooled_output.cpu().numpy()
        elif self.extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()
        elif self.extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        elif self.extraction_strategy == "s3e":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy,
                                     ignore_first_token=ignore_first_token,
                                     input_ids=input_ids, s3e_stats=self.s3e_stats)
        else:
            raise NotImplementedError

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.tokenized["tokens"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output, padding_mask, strategy, ignore_first_token, input_ids=None, s3e_stats=None):

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
        if strategy == "s3e":
            input_ids = input_ids.cpu().numpy()
            pooled_vecs = s3e_pooling(token_embs=token_vecs,
                                      token_ids=input_ids,
                                      token_weights=s3e_stats["token_weights"],
                                      centroids=s3e_stats["centroids"],
                                      token_to_cluster=s3e_stats["token_to_cluster"],
                                      svd_components=s3e_stats.get("svd_components", None),
                                      mask=padding_mask == 0)
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
            bert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
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
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


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
            albert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
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
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


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
            roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
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
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


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
            xlm_roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
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
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


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
            config = DistilBertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            distilbert.model = DistilBertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            distilbert.language = distilbert.model.config.language
        else:
            # Pytorch-transformer Style
            distilbert.model = DistilBertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            distilbert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
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
            xlnet.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
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
        # TODO verify that this is really doing correct pooling
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

class EmbeddingConfig():
    """
    Config for Word Embeddings Models.
    Necessary to work with Bert and other LM style functionality
    """
    def __init__(self,
                 name=None,
                 embeddings_filename=None,
                 vocab_filename=None,
                 vocab_size=None,
                 hidden_size=None,
                 language=None,
                 **kwargs):
        """
        :param name: Name of config
        :param embeddings_filename:
        :param vocab_filename:
        :param vocab_size:
        :param hidden_size:
        :param language:
        :param kwargs:
        """
        self.name = name
        self.embeddings_filename = embeddings_filename
        self.vocab_filename = vocab_filename
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.language = language
        if len(kwargs) > 0:
            logger.info(f"Passed unused params {str(kwargs)} to the EmbeddingConfig. Might not be a problem.")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class EmbeddingModel():
    """
    Embedding Model that combines
    - Embeddings
    - Config Object
    - Vocab
    Necessary to work with Bert and other LM style functionality
    """

    def __init__(self,
                 embedding_file,
                 config_dict,
                 vocab_file):
        """

        :param embedding_file: filename of embeddings. Usually in txt format, with the word and associated vector on each line
        :type embedding_file: str
        :param config_dict: dictionary containing config elements
        :type config_dict: dict
        :param vocab_file: filename of vocab, each line contains a word
        :type vocab_file: str
        """
        self.config = EmbeddingConfig(**config_dict)
        self.vocab = load_vocab(vocab_file)
        temp = wordembedding_utils.load_embedding_vectors(embedding_file=embedding_file, vocab=self.vocab)
        self.embeddings = torch.from_numpy(temp).float()
        assert "[UNK]" in self.vocab, "No [UNK] symbol in Wordembeddingmodel! Aborting"
        self.unk_idx = self.vocab["[UNK]"]

    def save(self,save_dir):
        # Save Weights
        save_name = Path(save_dir) / self.config.embeddings_filename
        embeddings = self.embeddings.cpu().numpy()
        with open(save_name, "w") as f:
            for w, vec in tqdm(zip(self.vocab, embeddings), desc="Saving embeddings", total=embeddings.shape[0]):
                f.write(w + " " + " ".join(["%.6f" % v for v in vec]) + "\n")
        f.close()

        # Save vocab
        save_name = Path(save_dir) / self.config.vocab_filename
        with open(save_name, "w") as f:
            for w in self.vocab:
                f.write(w + "\n")
        f.close()


    def resize_token_embeddings(self, new_num_tokens=None):
        # function is called as a vocab length validation inside FARM
        # fast way of returning an object with num_embeddings attribute (needed for some checks)
        # TODO add functionality to add words/tokens to a wordembeddingmodel after initialization
        temp = {}
        temp["num_embeddings"] = len(self.vocab)
        temp = DotMap(temp)
        return temp



class WordEmbedding_LM(LanguageModel):
    """
    A Language Model based only on word embeddings
    - Inside FARM, WordEmbedding Language Models must have a fixed vocabulary
    - Each (known) word in some text input is projected to its vector representation
    - Pooling operations can be applied for representing whole text sequences

    """

    def __init__(self):
        super(WordEmbedding_LM, self).__init__()
        self.model = None
        self.name = "WordEmbedding_LM"
        self.pooler = None


    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * a local path of a model trained via FARM ("some_dir/farm_model")
        * the name of a remote model on s3

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        wordembedding_LM = cls()
        if "farm_lm_name" in kwargs:
            wordembedding_LM.name = kwargs["farm_lm_name"]
        else:
            wordembedding_LM.name = pretrained_model_name_or_path
        # We need to differentiate between loading model from local or remote
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # local dir
            config = json.load(open(farm_lm_config,"r"))
            farm_lm_model = Path(pretrained_model_name_or_path) / config["embeddings_filename"]
            vocab_filename = Path(pretrained_model_name_or_path) / config["vocab_filename"]
            wordembedding_LM.model = EmbeddingModel(embedding_file=str(farm_lm_model), config_dict=config, vocab_file=str(vocab_filename))
            wordembedding_LM.language = config.get("language", None)
        else:
            # from remote or cache
            config_dict, resolved_vocab_file, resolved_model_file = wordembedding_utils.load_model(pretrained_model_name_or_path, **kwargs)
            model = EmbeddingModel(embedding_file=resolved_model_file,
                                   config_dict=config_dict,
                                   vocab_file=resolved_vocab_file)
            wordembedding_LM.model = model
            wordembedding_LM.language = model.config.language


        # taking the mean for getting the pooled representation
        # TODO: extend this to other pooling operations or remove
        wordembedding_LM.pooler = lambda x: torch.mean(x, dim=0)
        return wordembedding_LM

    def save(self, save_dir):
        """
        Save the model embeddings and its config file so that it can be loaded again.
        # TODO make embeddings trainable and save trained embeddings
        # TODO save model weights as pytorch model bin for more efficient loading and saving
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        #save model
        self.model.save(save_dir=save_dir)
        #save config
        self.save_config(save_dir=save_dir)


    def forward(self, input_ids, **kwargs,):
        """
        Perform the forward pass of the wordembedding model.
        This is just the mapping of words to their corresponding embeddings
        """
        sequence_output = []
        pooled_output = []
        # TODO do not use padding items in pooled output
        for sample in input_ids:
            sample_embeddings = []
            for index in sample:
                #if index != self.model.unk_idx:
                sample_embeddings.append(self.model.embeddings[index])
            sample_embeddings = torch.stack(sample_embeddings)
            sequence_output.append(sample_embeddings)
            pooled_output.append(self.pooler(sample_embeddings))

        sequence_output = torch.stack(sequence_output)
        pooled_output = torch.stack(pooled_output)
        m = nn.BatchNorm1d(pooled_output.shape[1])
        # use batchnorm for more stable learning
        # but disable it, if we have batch size of one (cannot compute batchnorm stats with only one sample)
        if pooled_output.shape[0] > 1:
            pooled_output = m(pooled_output)
        return sequence_output, pooled_output

    def trim_vocab(self, token_counts, processor, min_threshold):
        """ Remove embeddings for rare tokens in your corpus (< `min_threshold` occurrences) to reduce model size"""
        logger.info(f"Removing tokens with less than {min_threshold} occurrences from model vocab")
        new_vocab = OrderedDict()
        valid_tok_indices = []
        cnt = 0
        old_num_emb = self.model.embeddings.shape[0]
        for token, tok_idx in self.model.vocab.items():
            if token_counts.get(token, 0) >= min_threshold or token in ("[CLS]","[SEP]","[UNK]","[PAD]","[MASK]"):
                new_vocab[token] = cnt
                valid_tok_indices.append(tok_idx)
                cnt += 1

        self.model.vocab = new_vocab
        self.model.embeddings = self.model.embeddings[valid_tok_indices, :]

        # update tokenizer vocab in place
        processor.tokenizer.vocab = self.model.vocab
        processor.tokenizer.ids_to_tokens = OrderedDict()
        for k, v in processor.tokenizer.vocab.items():
            processor.tokenizer.ids_to_tokens[v] = k

        logger.info(f"Reduced vocab from {old_num_emb} to {self.model.embeddings.shape[0]}")

    def normalize_embeddings(self, zero_mean=True, pca_removal=False, pca_n_components=300, pca_n_top_components=10,
                             use_mean_vec_for_special_tokens=True, n_special_tokens=5):
        """ Normalize word embeddings as in https://arxiv.org/pdf/1808.06305.pdf
            (e.g. used for S3E Pooling of sentence embeddings)
            
        :param zero_mean: Whether to center embeddings via subtracting mean
        :type zero_mean: bool
        :param pca_removal: Whether to remove PCA components
        :type pca_removal: bool
        :param pca_n_components: Number of PCA components to use for fitting
        :type pca_n_components: int
        :param pca_n_top_components: Number of PCA components to remove
        :type pca_n_top_components: int
        :param use_mean_vec_for_special_tokens: Whether to replace embedding of special tokens with the mean embedding
        :type use_mean_vec_for_special_tokens: bool
        :param n_special_tokens: Number of special tokens like CLS, UNK etc. (used if `use_mean_vec_for_special_tokens`). 
                                 Note: We expect the special tokens to be the first `n_special_tokens` entries of the vocab.
        :type n_special_tokens: int
        :return: None
        """

        if zero_mean:
            logger.info('Removing mean from embeddings')
            # self.model.embeddings[:n_special_tokens, :] = torch.zeros((n_special_tokens, 300))
            mean_vec = torch.mean(self.model.embeddings, 0)
            self.model.embeddings = self.model.embeddings - mean_vec

            if use_mean_vec_for_special_tokens:
                self.model.embeddings[:n_special_tokens, :] = mean_vec

        if pca_removal:
            from sklearn.decomposition import PCA
            logger.info('Removing projections on top PCA components from embeddings (see https://arxiv.org/pdf/1808.06305.pdf)')
            pca = PCA(n_components=pca_n_components)
            pca.fit(self.model.embeddings.cpu().numpy())

            U1 = pca.components_
            explained_variance = pca.explained_variance_

            # Removing projections on top components
            PVN_dims = pca_n_top_components
            for emb_idx in tqdm(range(self.model.embeddings.shape[0]), desc="Removing projections"):
                for pca_idx, u in enumerate(U1[0:PVN_dims]):
                    ratio = (explained_variance[pca_idx] - explained_variance[PVN_dims]) / explained_variance[pca_idx]
                    self.model.embeddings[emb_idx] = self.model.embeddings[emb_idx] - ratio * np.dot(u.transpose(), self.model.embeddings[emb_idx]) * u


class Electra(LanguageModel):
    """
    ELECTRA is a new pre-training approach which trains two transformer models:
    the generator and the discriminator. The generator replaces tokens in a sequence,
    and is therefore trained as a masked language model. The discriminator, which is
    the model we're interested in, tries to identify which tokens were replaced by
    the generator in the sequence.

    The ELECTRA model here wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - Electra does not output the pooled_output. An additional pooler is initialized.

    """

    def __init__(self):
        super(Electra, self).__init__()
        self.model = None
        self.name = "electra"
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("google/electra-base-discriminator" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        electra = cls()
        if "farm_lm_name" in kwargs:
            electra.name = kwargs["farm_lm_name"]
        else:
            electra.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = ElectraConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            electra.model = ElectraModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            electra.language = electra.model.config.language
        else:
            # Transformers Style
            electra.model = ElectraModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            electra.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        config = electra.model.config

        # ELECTRA does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head.
        # Note: ELECTRA uses gelu as activation (BERT uses tanh instead)
        config.summary_last_dropout = 0
        config.summary_type = 'first'
        config.summary_activation = 'gelu'
        config.summary_use_proj = False
        electra.pooler = SequenceSummary(config)
        electra.pooler.apply(electra.model._init_weights)
        return electra

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the ELECTRA model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
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


class Camembert(Roberta):
    """
    A Camembert model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """
    def __init__(self):
        super(Camembert, self).__init__()
        self.model = None
        self.name = "camembert"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("camembert-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        camembert = cls()
        if "farm_lm_name" in kwargs:
            camembert.name = kwargs["farm_lm_name"]
        else:
            camembert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = CamembertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            camembert.model = CamembertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            camembert.language = camembert.model.config.language
        else:
            # Huggingface transformer Style
            camembert.model = CamembertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            camembert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return camembert


class DPRQuestionEncoder(LanguageModel):
    """
    A DPRQuestionEncoder model that wraps HuggingFace's implementation
    """

    def __init__(self):
        super(DPRQuestionEncoder, self).__init__()
        self.model = None
        self.name = "dpr_question_encoder"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("facebook/dpr-question_encoder-single-nq-base" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder
        :type pretrained_model_name_or_path: str
        """

        dpr_question_encoder = cls()
        if "farm_lm_name" in kwargs:
            dpr_question_encoder.name = kwargs["farm_lm_name"]
        else:
            dpr_question_encoder.name = pretrained_model_name_or_path

        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            dpr_config = transformers.DPRConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            dpr_question_encoder.model = transformers.DPRQuestionEncoder.from_pretrained(farm_lm_model, config=dpr_config, **kwargs)
            dpr_question_encoder.language = dpr_question_encoder.model.config.language
        else:
            model_type = AutoConfig.from_pretrained(pretrained_model_name_or_path).model_type
            if model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRQuestionEncoder model
                dpr_question_encoder.model = transformers.DPRQuestionEncoder.from_pretrained(
                    str(pretrained_model_name_or_path), **kwargs)
            else:
                # "from scratch": load weights from different architecture (e.g. bert) into DPRQuestionEncoder
                dpr_question_encoder.model = transformers.DPRQuestionEncoder(config=transformers.DPRConfig(**kwargs))
                dpr_question_encoder.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), **kwargs)
            dpr_question_encoder.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)

        return dpr_question_encoder

    def forward(
        self,
        query_input_ids,
        query_segment_ids,
        query_attention_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the DPRQuestionEncoder model.

        :param query_input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type query_input_ids: torch.Tensor
        :param query_segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type query_segment_ids: torch.Tensor
        :param query_attention_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :type query_attention_mask: torch.Tensor
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids=query_input_ids,
            token_type_ids=query_segment_ids,
            attention_mask=query_attention_mask,
            return_dict=True
        )
        if self.model.question_encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None

    def enable_hidden_states_output(self):
        self.model.question_encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.question_encoder.config.output_hidden_states = False


class DPRContextEncoder(LanguageModel):
    """
    A DPRContextEncoder model that wraps HuggingFace's implementation
    """

    def __init__(self):
        super(DPRContextEncoder, self).__init__()
        self.model = None
        self.name = "dpr_context_encoder"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("facebook/dpr-ctx_encoder-single-nq-base" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRContextEncoder
        :type pretrained_model_name_or_path: str
        """

        dpr_context_encoder = cls()
        if "farm_lm_name" in kwargs:
            dpr_context_encoder.name = kwargs["farm_lm_name"]
        else:
            dpr_context_encoder.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            dpr_config = transformers.DPRConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            dpr_context_encoder.model = transformers.DPRContextEncoder.from_pretrained(farm_lm_model, config=dpr_config, **kwargs)
            dpr_context_encoder.language = dpr_context_encoder.model.config.language
        else:
            # Pytorch-transformer Style
            model_type = AutoConfig.from_pretrained(pretrained_model_name_or_path).model_type
            if model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRContextEncoder model
                dpr_context_encoder.model = transformers.DPRContextEncoder.from_pretrained(
                    str(pretrained_model_name_or_path), **kwargs)
            else:
                # "from scratch": load weights from different architecture (e.g. bert) into DPRContextEncoder
                dpr_context_encoder.model = transformers.DPRContextEncoder(config=transformers.DPRConfig(**kwargs))
                dpr_context_encoder.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), **kwargs)
            dpr_context_encoder.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)

        return dpr_context_encoder

    def forward(
        self,
        passage_input_ids,
        passage_segment_ids,
        passage_attention_mask,
        **kwargs,
    ):
        """
        Perform the forward pass of the DPRContextEncoder model.

        :param passage_input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len]
        :type passage_input_ids: torch.Tensor
        :param passage_segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len]
        :type passage_segment_ids: torch.Tensor
        :param passage_attention_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size,  number_of_hard_negative_passages, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        max_seq_len = passage_input_ids.shape[-1]
        passage_input_ids = passage_input_ids.view(-1, max_seq_len)
        passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)
        passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)
        output_tuple = self.model(
            input_ids=passage_input_ids,
            token_type_ids=passage_segment_ids,
            attention_mask=passage_attention_mask,
            return_dict=True
        )
        if self.model.ctx_encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None

    def enable_hidden_states_output(self):
        self.model.ctx_encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.ctx_encoder.config.output_hidden_states = False
