import ast
import datetime
import json
import os
from dotmap import DotMap

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.modeling import BertPreTrainingHeads


class PredictionHead(nn.Module):
    """ Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions. """

    @classmethod
    def load(cls, load_dir):
        # TODO: Maybe we want to initialize at higher so that switching in a new config can give us a whole new class of ph
        filepath = os.path.join(load_dir, "prediction_head_config.json")
        with open(filepath) as file:
            config = json.load(file)
        return cls(**config)

    def save_config(self, save_dir):
        output_config_file = os.path.join(save_dir, "prediction_head_config.json")
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def checkpoint(self, save_dir, step="X"):

        # Save a trained model, configuration and tokenizer
        model_to_save = (
            self.module if hasattr(self, "module") else self
        )  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(
            save_dir, "prediction_head_{}.bin".format(step)
        )

        torch.save(model_to_save.state_dict(), output_model_file)

    # TODO Should these be Abstract methods? i.e. enforce that they are implemented in the child class
    def logits_to_loss(self, logits, labels):
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        raise NotImplementedError()


class TextClassificationHead(PredictionHead):
    def __init__(self, layer_dims, class_weights=None, **kwargs):
        super(TextClassificationHead, self).__init__()
        self.layer_dims_list = ast.literal_eval(str(layer_dims))
        self.feed_forward = FeedForwardBlock(self.layer_dims_list)
        self.generate_config()
        self.num_labels = self.layer_dims_list[-1]
        self.ph_output_type = "per_sequence"

        # Todo do we still need to do this?
        if class_weights:
            self.balanced_weights = nn.Parameter(
                torch.tensor(class_weights), requires_grad=False
            )
            self.loss_fct = CrossEntropyLoss(
                weight=self.balanced_weights, reduction="none"
            )
        else:
            self.loss_fct = CrossEntropyLoss(reduction="none")

    def generate_config(self):
        self.config = {
            "type": type(self).__name__,
            "last_initialized": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer_dims": str(self.layer_dims_list),
        }

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, label_ids, **kwargs):
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.argmax(1)
        # TODO: Two are returned because token level classification currently returns label ids as well. This should be changed
        return None, preds


class TokenClassificationHead(PredictionHead):
    def __init__(self, layer_dims, **kwargs):
        super(TokenClassificationHead, self).__init__()
        self.layer_dims_list = ast.literal_eval(str(layer_dims))
        self.feed_forward = FeedForwardBlock(self.layer_dims_list)
        self.generate_config()
        self.num_labels = self.layer_dims_list[-1]
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.ph_output_type = "per_token"

    def generate_config(self):
        self.config = {
            "type": type(self).__name__,
            "last_initialized": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer_dims": str(self.layer_dims_list),
        }

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(
        self, logits, label_ids, initial_mask, attention_mask=None, **kwargs
    ):
        # Todo: should we be applying initial mask here? Loss is currently calculated even on non initial tokens
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)
        return loss

    def logits_to_preds(self, logits, initial_mask, label_map, label_ids, **kwargs):

        preds_word_all = []
        labels_word_all = []

        preds_tokens = torch.argmax(logits, dim=2)

        preds_token = preds_tokens.detach().cpu().numpy()
        # used to be: input_mask = input_mask.detach().cpu().numpy()
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


class BertLanguageModelHead(PredictionHead):
    """ Masked Language Model with NextSentence Prediction"""

    def __init__(self, embeddings, hidden_size, hidden_act="gelu", **kwargs):
        super(BertLanguageModelHead, self).__init__()

        # self.bert = BertModel(config)
        config = {"hidden_size": hidden_size, "hidden_act": hidden_act}
        config = DotMap(config, _dynamic=False)
        embeddings_weights = embeddings.word_embeddings.weight
        self.multihead = BertPreTrainingHeads(config, embeddings_weights)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        # TODO change dummy value to real one
        self.num_labels = embeddings_weights.shape[0]  # vocab size
        # TODO Check if weight init needed!
        # self.apply(self.init_bert_weights)

        self.generate_config()

    def generate_config(self):
        self.config = {
            "type": type(self).__name__,
            "last_initialized": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # "layer_dims": str(self.layer_dims),
        }

    def forward(self, X):
        lm_logits, next_sentence_logits = self.multihead(X[0], X[1])
        return [lm_logits, next_sentence_logits]

    def logits_to_loss(self, logits, label_ids, is_next, **kwargs):
        assert len(logits) == 2
        masked_lm_loss = self.loss_fct(
            logits[0].view(-1, self.num_labels), label_ids.view(-1)
        )
        next_sentence_loss = self.loss_fct(logits[1].view(-1, 2), is_next.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        total_loss = total_loss.view(1, 1)
        return total_loss

    def logits_to_preds(self, logits, is_next, **kwargs):
        # TODO does logits shape really allow just argmax here?
        lm_preds = logits[0].argmax(1)
        next_sentence_preds = logits[1].argmax(1)
        # TODO return lm_preds for eval as well
        # TODO: Two are returned because token level classification currently returns label ids as well. This should be changed
        return is_next, next_sentence_preds


class FeedForwardBlock(nn.Module):
    """ A feed forward neural network of variable depth and width. """

    def __init__(self, layer_dims, **kwargs):
        # Todo: Consider having just one input argument
        super(FeedForwardBlock, self).__init__()

        # If read from config the input will be string
        n_layers = len(layer_dims) - 1
        layers_all = []
        # TODO: IS this needed?
        self.output_size = layer_dims[-1]

        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits
