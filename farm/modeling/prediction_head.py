import ast
import datetime
import json
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


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


class SeqClassificationHead(PredictionHead):
    def __init__(self, layer_dims, balanced_weights=None, **kwargs):
        super(SeqClassificationHead, self).__init__()
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(layer_dims)
        self.generate_config()
        self.num_labels = layer_dims[-1]
        if balanced_weights:
            self.balanced_weights = nn.Parameter(
                torch.tensor(balanced_weights), requires_grad=False
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
            "layer_dims": str(self.layer_dims),
        }

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        return self.loss_fct(logits, labels.view(-1))

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.argmax(1)
        # TODO: Two are returned because token level classification currently returns label ids as well. This should be changed
        return None, preds


class NERClassificationHead(PredictionHead):
    def __init__(self, layer_dims, **kwargs):
        super(NERClassificationHead, self).__init__()
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(layer_dims)
        self.generate_config()
        self.num_labels = layer_dims[-1]
        self.loss_fct = CrossEntropyLoss(reduction="none")

    def generate_config(self):
        self.config = {
            "type": type(self).__name__,
            "last_initialized": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer_dims": str(self.layer_dims),
        }

    def forward(self, X):
        logits = self.feed_forward(X)
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
        # input_mask = input_mask.detach().cpu().numpy()
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


class FeedForwardBlock(nn.Module):
    """ A feed forward neural network of variable depth and width. """

    def __init__(self, layer_dims, **kwargs):
        # Todo: Consider having just one input argument
        super(FeedForwardBlock, self).__init__()

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

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits
