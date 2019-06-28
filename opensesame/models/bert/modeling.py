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
import datetime
import json
import ast


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, Module
from pytorch_pretrained_bert.modeling import BertConfig, BertLayerNorm, load_tf_weights_in_bert, BertEncoder, \
    BertEmbeddings, BertPooler, BertPreTrainingHeads, BertOnlyMLMHead, BertOnlyNSPHead, BertPreTrainedModel

from opensesame.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
    'bert-base-cased-de-v0-1': "s3://int-models-bert/bert-base-cased-de-v0-1/bert-base-cased-de-v0-1.tar.gz",
    'bert-base-cased-de-1a-start': "s3://int-models-bert/bert-base-cased-de-1a-start/bert-base-cased-de-1a-start.tar.gz",
    'bert-base-cased-de-1a-10k': "s3://int-models-bert/bert-base-cased-de-1a-10k/bert-base-cased-de-1a-10k.tar.gz",
    'bert-base-cased-de-1a-20k': "s3://int-models-bert/bert-base-cased-de-1a-20k/bert-base-cased-de-1a-20k.tar.gz",
    'bert-base-cased-de-1a-50k': "s3://int-models-bert/bert-base-cased-de-1a-50k/bert-base-cased-de-1a-50k.tar.gz",
    'bert-base-cased-de-1a-end': "s3://int-models-bert/bert-base-cased-de-1a-end/bert-base-cased-de-1a-end.tar.gz",
    'bert-base-cased-de-1b-end': "s3://int-models-bert/bert-base-cased-de-1b-end/bert-base-cased-de-1b-end.tar.gz",
    'bert-base-cased-de-1b-best': "s3://int-models-bert/bert-base-cased-de-1b-best/bert-base-cased-de-1b-best.tar.gz",
    'bert-base-cased-de-2a-end': "s3://int-models-bert/bert-base-cased-de-2a-end/bert-base-cased-de-2a-end.tar.gz",
    'bert-base-cased-de-2b-end': "s3://int-models-bert/bert-base-cased-de-2b-end/bert-base-cased-de-2b-end.tar.gz",
}

BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'


class FARMModel(Module):
    def __init__(self, language_model, prediction_head):
        super(FARMModel, self).__init__()
        self.language_model = language_model
        self.prediction_head = prediction_head

    def logits_to_loss(self, logits, labels):
        raise NotImplementedError()

    def logits_to_pred(self, logits):
        raise NotImplementedError()

    def save(self, save_dir):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_dir):
        raise NotImplementedError()


class LanguageModel(Module):
    def freeze(self, layers):
        raise NotImplementedError()

    def unfreeze(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_dir):
        raise NotImplementedError()

    def save_config(self, save_dir):
        raise NotImplementedError()

    def checkpoint(self, save_dir):
        """
        Todo: How do we want to implement this? Should probably have a switch to turn off and on checkpointing
        People will want to checkpoint if finetuning but not if LM is frozen
        """
        raise NotImplementedError()


class PredictionHead(Module):
    @classmethod
    def load(cls, load_dir):
        raise NotImplementedError()

    def save(self, save_dir):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_dir):
        # TODO: Maybe we want to initialize at higher so that switching in a new config can give us a whole new class of ph
        filepath = os.path.join(load_dir, "prediction_head_config.json")
        with open(filepath) as file:
            config = json.load(file)
        return cls(**config)


class FeedForwardFarm(PredictionHead):
    def __init__(self,
                 layer_dims,
                 **kwargs):
        # Todo: Consider having just one input argument
        super(FeedForwardFarm, self).__init__()

        # If read from config the input will be string
        self.layer_dims = ast.literal_eval(str(layer_dims))
        n_layers = len(self.layer_dims) - 1
        layers_all = []
        # TODO: IS this needed?
        self.output_size = self.layer_dims[-1]

        for i in range(n_layers):
            size_in = self.layer_dims[i]
            size_out = self.layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

        self.generate_config()


    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def save_config(self, save_dir):
        output_config_file = os.path.join(save_dir, "prediction_head_config.json")
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def checkpoint(self, save_dir, step="X"):

        # Save a trained model, configuration and tokenizer
        model_to_save = self.module if hasattr(self, 'module') else self  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_dir, "prediction_head_{}.bin".format(step))

        torch.save(model_to_save.state_dict(), output_model_file)

    def generate_config(self):
        self.config = {"type": type(self).__name__,
                       "last_initialized": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "layer_dims": str(self.layer_dims)}

class BertFarm(LanguageModel):
    def __init__(self, pretrained_model_name_or_path):
        super(BertFarm, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers=False):
        return self.bert(input_ids,
                         token_type_ids,
                         attention_mask,
                         output_all_encoded_layers=False)

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        return BertFarm(pretrained_model_name_or_path)

    def save_config(self, save_dir):
        # TODO: Maybe we want to initialize at higher so that switching in a new config can give us a whole new class of lm
        output_config_file = os.path.join(save_dir, "language_model_config.json")
        self.bert.config.to_json_file(output_config_file)


class BertSeqFarm(FARMModel):
    def __init__(self,
                 language_model,
                 prediction_head,
                 embeds_dropout_prob,
                 balanced_weights=None):

        super(BertSeqFarm, self).__init__(language_model, prediction_head)

        self.num_labels = prediction_head.output_size

        self.dropout = nn.Dropout(embeds_dropout_prob)

        # needs to be a parameter for distributed setting
        # This is messy, can we do this differently?
        if balanced_weights:
            self.balanced_weights = torch.nn.Parameter(torch.tensor(balanced_weights), requires_grad=False)
            self.loss_fct = CrossEntropyLoss(weight=self.balanced_weights, reduction="none")
        else:
            self.loss_fct = CrossEntropyLoss(reduction="none")

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None):

        _, pooled_output = self.language_model(input_ids,
                                               token_type_ids,
                                               attention_mask,
                                               output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)

        # TODO: is self.num_labels needed here? Doesn't prediction head already do this?
        logits = self.prediction_head(pooled_output).view(-1, self.num_labels)

        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        return self.loss_fct(logits, labels.view(-1))

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.argmax(1)
        # TODO: Two are returned because token level classification currently returns label ids as well. This should be changed
        return None, preds


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
                ))
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
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

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
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                for member in archive.getmembers():
                    if member.isreg():  # skip if the TarInfo is not files
                        member.name = os.path.basename(member.name)  # remove the path by reset it
                        archive.extract(member, tempdir)  # extract
                # archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
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
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
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
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
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
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels, balanced_weights=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        if balanced_weights:
            assert self.num_labels == len(balanced_weights)
            # needs to be a parameter for distributed setting
            self.balanced_weights = torch.nn.Parameter(torch.tensor(balanced_weights), requires_grad=False)
        else:
            self.balanced_weights = None
        self.loss_fct = CrossEntropyLoss(weight=self.balanced_weights, reduction="none")
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).view(-1, self.num_labels)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        return self.loss_fct(logits, labels.view(-1))

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.argmax(1)
        # preds = np.argmax(logits, axis=1)
        # TODO: Two are returned because token level classification currently returns label ids as well. This should be changed
        return None, preds


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

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
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # TODO: In the other models, CrossEntropyLoss is expected to return per sample loss (using the reduction = none argument)
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def logits_to_loss(self, logits, labels, initial_mask, attention_mask=None):
        # Todo: should we be applying initial mask here? Loss is currently calculated even on non initial tokens
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)
        return loss

    def logits_to_preds(self, logits, input_mask, initial_mask, label_map, label_ids):

        preds_word_all = []
        labels_word_all = []

        preds_tokens = torch.argmax(logits, dim=2)

        preds_token = preds_tokens.detach().cpu().numpy()
        input_mask = input_mask.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        for idx, im in enumerate(initial_mask):
            preds_t = preds_token[idx]
            labels_t = label_ids[idx]

            # Get labels and predictions for just the word initial tokens
            labels_word_id = self.initial_token_only(labels_t, initial_mask=im)
            preds_word_id = self.initial_token_only(preds_t, initial_mask=im)

            labels_word = [label_map[lwi] for lwi in labels_word_id]
            preds_word = [label_map[pwi] for pwi in preds_word_id]

            preds_word_all.append(preds_word)
            labels_word_all.append(labels_word)

        return labels_word_all, preds_word_all

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret
