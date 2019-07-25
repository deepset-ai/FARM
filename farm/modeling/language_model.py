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

import torch
from torch import nn

logger = logging.getLogger(__name__)

from pytorch_transformers.modeling_bert import BertModel, BertConfig


class LanguageModel(nn.Module):
    """
    Takes a tokenized sentence as input and returns vectors that represents the input semantically.
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
    def load(cls, load_dir):
        config_file = os.path.join(load_dir, "language_model_config.json")
        if os.path.exists(config_file):
            # it's a local directory
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(load_dir)
        else:
            # it's a model name which we try to resolve from s3. for now only works for bert models
            language_model = cls.subclasses["Bert"].load(load_dir)
        return language_model

    def freeze(self, layers):
        raise NotImplementedError()

    def unfreeze(self):
        raise NotImplementedError()

    def save_config(self, save_dir):
        raise NotImplementedError()

    def save(self, save_dir):
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

    def formatted_preds(self, input_ids, samples, extraction_strategy="pooled", **kwargs):
        sequence_output, pooled_output = self.forward(input_ids, output_all_encoded_layers=False, **kwargs)

        if extraction_strategy == "pooled":
            vecs = pooled_output.cpu().numpy()
        elif extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()
        else:
            raise NotImplementedError

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.tokenized["tokens"]
            pred["vec"] = vec
            preds.append(pred)

        return preds


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
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = os.path.join(pretrained_model_name_or_path, "language_model_config.json")
        if os.path.exists(farm_lm_config):
            # FARM style
            bert_config = BertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = os.path.join(pretrained_model_name_or_path, "language_model.bin")
            bert.model = BertModel.from_pretrained(farm_lm_model, config = bert_config)
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
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        sequence_output, pooled_output = output_tuple[0], output_tuple[1]
        return sequence_output, pooled_output

    def save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "language_model_config.json")
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)