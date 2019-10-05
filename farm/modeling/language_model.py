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
""" Acknowledgements: Many of the modeling parts here come from the great pytorch-transformers repository: https://github.com/huggingface/pytorch-transformers.
Thansk for the great work! """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import json
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

from transformers.modeling_bert import BertModel, BertConfig


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

    def forward(self):
        raise NotImplementedError

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Load a pretrained language model either by 1) specifying its name and downloading it or 2) pointing
        to the directory it is saved in.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name. Choose from:

              * `bert-base-uncased`
              * `bert-large-uncased`
              * `bert-base-cased`
              * `bert-large-cased`
              * `bert-base-multilingual-uncased`
              * `bert-base-multilingual-cased`
              * `bert-base-chinese`
              * `bert-base-german-cased`

        :type pretrained_model_name_or_path: str
        """
        config_file = os.path.join(pretrained_model_name_or_path, "language_model_config.json")
        if os.path.exists(config_file):
            # it's a local directory
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path)
        else:
            # it's a model name which we try to resolve from s3. for now only works for bert models
            language_model = cls.subclasses["Bert"].load(pretrained_model_name_or_path)

        assert language_model is not None

        return language_model

    def freeze(self, layers):
        """ To be implemented"""
        raise NotImplementedError()

    def unfreeze(self):
        """ To be implemented"""
        raise NotImplementedError()

    def save_config(self, save_dir):
        """ To be implemented"""
        raise NotImplementedError()

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        # Save Weights
        save_name = os.path.join(save_dir, "language_model.bin")
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
                "We guess it's an *ENGLISH* model ... \n"
                "If not: Init the language model by supplying the 'language' param.\n"
                "Example: Bert.load('my_mysterious_model_name', language='de')"
            )
        elif len(matches) > 1:
            raise ValueError(
                "Could not automatically detect from language model name what language it is.\n"
                f"Found multiple matches: {matches}\n"
                "Please init the language model by manually supplying the 'language' as a parameter.\n"
                "Example: Bert.load('my_mysterious_model_name', language='de')"
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
    """ A BERT model (https://arxiv.org/abs/1810.04805) that wraps the HuggingFace's implementation
    (https://github.com/huggingface/pytorch-pretrained-BERT) to fit the LanguageModel class. """

    def __init__(self):
        super(Bert, self).__init__()
        self.model = None
        self.name = "bert"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None):
        bert = cls()
        bert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = os.path.join(pretrained_model_name_or_path, "language_model_config.json")
        if os.path.exists(farm_lm_config):
            # FARM style
            bert_config = BertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = os.path.join(pretrained_model_name_or_path, "language_model.bin")
            bert.model = BertModel.from_pretrained(farm_lm_model, config=bert_config)
            bert.language = bert.model.config.language
        else:
            # Pytorch-transformer Style
            bert.model = BertModel.from_pretrained(pretrained_model_name_or_path)
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

    def save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "language_model_config.json")
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)