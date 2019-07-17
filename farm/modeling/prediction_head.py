import ast
import datetime
import json
import os
from dotmap import DotMap
import random

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import logging

from pytorch_pretrained_bert.modeling import BertLMPredictionHead

logger = logging.getLogger(__name__)


class PredictionHead(nn.Module):
    """ Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions. """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create(cls, prediction_head_name, layer_dims, class_weights=None):
        # TODO make we want to make this more generic. Class weights is not relevant for all heads.
        # We could again use **kwargs
        return cls.subclasses[prediction_head_name](
            layer_dims=layer_dims, class_weights=class_weights
        )

    def save_config(self, save_dir, head_num=0):
        output_config_file = os.path.join(
            save_dir, f"prediction_head_{head_num}_config.json"
        )
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def save(self, save_dir, head_num=0):
        output_model_file = os.path.join(save_dir, f"prediction_head_{head_num}.bin")
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        self.config = {
            k: v
            for k, v in self.__dict__.items()
            if (type(v) in [str, int, bool, float])
        }
        self.config.update({"name": self.__class__.__name__})

    @classmethod
    def load(cls, model_file, config_file):
        config = json.load(open(config_file))
        # TODO make this more generic for other heads with more attributes
        # e.g parse all args from config and feed them as **kwargs to subclasses constructor
        prediction_head = cls.subclasses[config["name"]](
            layer_dims=config["layer_dims_str"]
        )
        logger.info("Loading prediction head from {}".format(model_file))
        prediction_head.load_state_dict(torch.load(model_file))
        return prediction_head

    def logits_to_loss(self, logits, labels):
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        raise NotImplementedError()

    def prepare_labels(self, label_map, **kwargs):
        raise NotImplementedError()


class TextClassificationHead(PredictionHead):
    def __init__(
        self,
        layer_dims,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        **kwargs,
    ):
        super(TextClassificationHead, self).__init__()
        # TODO MP I think it would be nicer here to pass hidden_dim and num_labels and construct layer_dims from it.
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims_str = layer_dims
        self.layer_dims_list = ast.literal_eval(str(layer_dims))
        self.feed_forward = FeedForwardBlock(self.layer_dims_list)
        self.num_labels = self.layer_dims_list[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"

        # Todo do we still need to do this?
        if class_weights:
            self.balanced_weights = nn.Parameter(
                torch.tensor(class_weights), requires_grad=False
            )
            self.loss_fct = CrossEntropyLoss(
                weight=self.balanced_weights,
                reduction=loss_reduction,
                ignore_index=loss_ignore_index,
            )
        else:
            self.loss_fct = CrossEntropyLoss(
                reduction=loss_reduction, ignore_index=loss_ignore_index
            )
        self.generate_config()

    @classmethod
    def load(cls, config_file, model_file):
        config = json.load(open(config_file))
        # TODO make this more generic for other heads with more attributes
        prediction_head = cls(config["layer_dims_str"])
        logger.info("Loading prediction head from {}".format(model_file))
        prediction_head.load_state_dict(torch.load(model_file))
        return prediction_head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, label_ids, **kwargs):
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_probs(self, logits, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, label_map, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [label_map[x] for x in pred_ids]
        return preds

    def prepare_labels(self, label_map, label_ids, **kwargs):
        label_ids = label_ids.cpu().numpy()
        labels = [label_map[int(x)] for x in label_ids]
        return labels

    def formatted_preds(self, logits, label_map, samples, **kwargs):
        preds = self.logits_to_preds(logits, label_map)
        probs = self.logits_to_probs(logits)
        contexts = [sample.clear_text["text"] for sample in samples]

        assert len(preds) == len(probs) == len(contexts)

        res = {"task": "text_classification", "prediction": []}
        for pred, prob, context in zip(preds, probs, contexts):
            res["prediction"].append(
                {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            )
        return res


class TokenClassificationHead(PredictionHead):
    def __init__(self, layer_dims, **kwargs):
        super(TokenClassificationHead, self).__init__()
        # TODO having layer_dims as str and list here is not pretty. I would rather have the string only in load() and save()
        self.layer_dims_str = layer_dims
        self.layer_dims_list = ast.literal_eval(str(layer_dims))
        self.feed_forward = FeedForwardBlock(self.layer_dims_list)
        self.num_labels = self.layer_dims_list[-1]
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.ph_output_type = "per_token"
        self.model_type = "token_classification"
        self.generate_config()

    @classmethod
    def load(cls, config, checkpoint_file):
        prediction_head = cls(config["layer_dims_str"])
        logger.info("Loading prediction head from {}".format(checkpoint_file))
        prediction_head.load_state_dict(torch.load(checkpoint_file))
        return prediction_head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(
        self, logits, label_ids, initial_mask, padding_mask=None, **kwargs
    ):
        # Todo: should we be applying initial mask here? Loss is currently calculated even on non initial tokens
        active_loss = padding_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)
        return loss

    def logits_to_preds(self, logits, initial_mask, label_map, **kwargs):
        preds_word_all = []
        preds_tokens = torch.argmax(logits, dim=2)
        preds_token = preds_tokens.detach().cpu().numpy()
        # used to be: padding_mask = padding_mask.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()

        for idx, im in enumerate(initial_mask):
            preds_t = preds_token[idx]
            # Get labels and predictions for just the word initial tokens
            preds_word_id = self.initial_token_only(preds_t, initial_mask=im)
            preds_word = [label_map[pwi] for pwi in preds_word_id]
            preds_word_all.append(preds_word)
        return preds_word_all

    def logits_to_probs(self, logits, initial_mask, **kwargs):
        # get per token probs
        softmax = torch.nn.Softmax(dim=2)
        token_probs = softmax(logits)
        token_probs = torch.max(token_probs, dim=2)[0]
        token_probs = token_probs.cpu().numpy()

        # convert to per word probs
        all_probs = []
        initial_mask = initial_mask.detach().cpu().numpy()
        for idx, im in enumerate(initial_mask):
            probs_t = token_probs[idx]
            probs_words = self.initial_token_only(probs_t, initial_mask=im)
            all_probs.append(probs_words)
        return all_probs

    def prepare_labels(self, label_map, label_ids, initial_mask, **kwargs):
        labels_all = []
        label_ids = label_ids.cpu().numpy()
        for label_ids_one_sample, initial_mask_one_sample in zip(
            label_ids, initial_mask
        ):
            label_ids = self.initial_token_only(
                label_ids_one_sample, initial_mask_one_sample
            )
            labels = [label_map[l] for l in label_ids]
            labels_all.append(labels)
        return labels_all

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret

    def formatted_preds(
        self, logits, label_map, tokenizer, initial_mask, input_ids, **kwargs
    ):
        preds = self.logits_to_preds(logits, initial_mask, label_map)
        probs = self.logits_to_probs(logits, initial_mask)

        # convert input_ids back to words
        vocab = {v: k for k, v in tokenizer.vocab.items()}
        input_ids = input_ids.cpu().numpy().tolist()
        words = []
        for seq in input_ids:
            tokens_seq = [vocab[t] for t in seq]
            words_seq = []
            word = ""
            for tok in tokens_seq:
                if "##" in tok:
                    word += tok.replace("##", "")
                elif tok not in ("[SEP]", "[CLS]", "[PAD]", "[unused3001]"):
                    if word != "":
                        words_seq.append(word)
                    word = tok
            words.append(words_seq)

        # contexts = [sample.clear_text["text"] for sample in samples]

        # assert len(preds) == len(probs) == len(words)
        #
        res = {"task": "ner", "prediction": []}
        for preds_seq, probs_seq, words_seq in zip(preds, probs, words):
            for pred, prob, word in zip(preds_seq, probs_seq, words_seq):
                rand_start = random.randint(0, len(words_seq))
                res["prediction"].append(
                    {
                        "start": rand_start,
                        "end": rand_start + (len(word)),
                        "context": f"{word}",
                        "label": f"{pred}",
                        "probability": prob,
                    }
                )
        return res


class BertLMHead(PredictionHead):
    def __init__(self, embeddings, hidden_size, hidden_act="gelu", **kwargs):
        super(BertLMHead, self).__init__()

        config = {"hidden_size": hidden_size, "hidden_act": hidden_act}
        config = DotMap(config, _dynamic=False)
        embeddings_weights = embeddings.word_embeddings.weight

        self.model = BertLMPredictionHead(config, embeddings_weights)
        self.loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.num_labels = embeddings_weights.shape[0]  # vocab size
        # TODO Check if weight init needed!
        # self.apply(self.init_bert_weights)
        self.ph_output_type = "per_token"
        self.generate_config()

    def forward(self, X):
        lm_logits = self.model(X)
        return lm_logits

    def logits_to_loss(self, logits, lm_label_ids, **kwargs):
        batch_size = lm_label_ids.shape[0]
        masked_lm_loss = self.loss_fct(
            logits.view(-1, self.num_labels), lm_label_ids.view(-1)
        )
        per_sample_loss = masked_lm_loss.view(-1, batch_size).mean(dim=0)
        return per_sample_loss

    def logits_to_preds(self, logits, label_map, lm_label_ids, **kwargs):
        logits = logits.cpu().numpy()
        lm_label_ids = lm_label_ids.cpu().numpy()
        lm_preds_ids = logits.argmax(2)
        # apply mask to get rid of predictions for non-masked tokens
        assert lm_preds_ids.shape == lm_label_ids.shape
        lm_preds_ids[lm_label_ids == -1] = -1
        lm_preds_ids = lm_preds_ids.tolist()
        preds = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for pred_ids_for_sequence in lm_preds_ids:
            preds.append(
                [label_map[int(x)] for x in pred_ids_for_sequence if int(x) != -1]
            )
        return preds

    def prepare_labels(self, label_map, lm_label_ids, **kwargs):
        label_ids = lm_label_ids.cpu().numpy().tolist()
        labels = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for ids_for_sequence in label_ids:
            labels.append([label_map[int(x)] for x in ids_for_sequence if int(x) != -1])
        return labels


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
