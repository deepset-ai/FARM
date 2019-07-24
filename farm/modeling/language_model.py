# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import shutil
import tarfile
import tempfile
import json

import torch
from pytorch_pretrained_bert.modeling import (
    BertConfig,
    BertLayerNorm,
    load_tf_weights_in_bert,
    BertEncoder,
    BertEmbeddings,
    BertPooler,
    BertPreTrainedModel,
)
from torch import nn

from farm.file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
    "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased.tar.gz",
}

CONFIG_NAMES = ["bert_config.json", "language_model_config.json"]
WEIGHTS_NAMES = ["pytorch_model.bin", "language_model.bin"]
TF_WEIGHTS_NAME = "model.ckpt"


class LanguageModel(nn.Module):
    """ Takes a tokenized sentence as input and returns vectors that represents the input semantically. """

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
        bert.model = BertModel.from_pretrained(pretrained_model_name_or_path)
        # set language by checking in bert.model.config
        if hasattr(bert.model.config, "language"):
            # it is our FARM style config
            language = bert.model.config.language
        else:
            # it is a huggingface style config
            language = cls._infer_language_from_name(pretrained_model_name_or_path)
        bert.language = language
        return bert

    def forward(
        self,
        input_ids,
        segment_ids,
        padding_mask,
        output_all_encoded_layers=False,
        **kwargs,
    ):
        return self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )

    def save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "language_model_config.json")
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)


# TODO: This class is huge and we only have it here because we need to load our models from S3. Is there a better way to do this?
# The above PRETRAINED_MODEL_ARCHIVE_MAP cannot be used inside pytorch huggingface bert package
class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get("state_dict", None)
        kwargs.pop("state_dict", None)
        cache_dir = kwargs.get("cache_dir", None)
        kwargs.pop("cache_dir", None)
        from_tf = kwargs.get("from_tf", False)
        kwargs.pop("from_tf", None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file,
                )
            )
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info(
                "loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file
                )
            )
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info(
                "extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir
                )
            )
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                for member in archive.getmembers():
                    if member.isreg():  # skip if the TarInfo is not files
                        member.name = os.path.basename(
                            member.name
                        )  # remove the path by reset it
                        archive.extract(member, tempdir)  # extract
                # archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        for config_name in CONFIG_NAMES:
            config_file = os.path.join(serialization_dir, config_name)
            if os.path.exists(config_file):
                break
            # # Backward compatibility with old naming format
            # config_file = os.path.join(serialization_dir, BERT_CONFIG_NAMES)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            for weights_name in WEIGHTS_NAMES:
                weights_path = os.path.join(serialization_dir, weights_name)
                if os.path.exists(weights_path):
                    break
            state_dict = torch.load(weights_path, map_location="cpu")
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(
            s.startswith("bert.") for s in state_dict.keys()
        ):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.warning(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.warning(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    padding_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, padding_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
