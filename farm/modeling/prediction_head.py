import itertools
import json
import logging
import os
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
import tqdm

import torch
from transformers.modeling_bert import BertForPreTraining, BertLayerNorm, ACT2FN, BertForQuestionAnswering

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from farm.data_handler.utils import is_json
from farm.utils import convert_iob_to_simple_tags, decode_squad_id

logger = logging.getLogger(__name__)


class PredictionHead(nn.Module):
    """ Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions. """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create(cls, prediction_head_name, layer_dims, class_weights=None):
        """
        Create subclass of Prediction Head.

        :param prediction_head_name: Classname (exact string!) of prediction head we want to create
        :type prediction_head_name: str
        :param layer_dims: describing the feed forward block structure, e.g. [768,2]
        :type layer_dims: List[Int]
        :param class_weights: The loss weighting to be assigned to certain label classes during training.
           Used to correct cases where there is a strong class imbalance.
        :type class_weights: list[Float]
        :return: Prediction Head of class prediction_head_name
        """
        # TODO make we want to make this more generic.
        #  1. Class weights is not relevant for all heads.
        #  2. Layer weights impose FF structure, maybe we want sth else later
        # Solution: We could again use **kwargs
        return cls.subclasses[prediction_head_name](
            layer_dims=layer_dims, class_weights=class_weights
        )

    def save_config(self, save_dir, head_num=0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :type save_dir: str
        :param head_num: Which head to save
        :type head_num: int
        """
        output_config_file = os.path.join(
            save_dir, f"prediction_head_{head_num}_config.json"
        )
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def save(self, save_dir, head_num=0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :type save_dir: str
        :param head_num: which head to save
        :type head_num: int
        """
        output_model_file = os.path.join(save_dir, f"prediction_head_{head_num}.bin")
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if is_json(value) and key[0] != "_":
                config[key] = value
        config["name"] = self.__class__.__name__
        self.config = config

    @classmethod
    def load(cls, config_file):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :type config_file: str
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        config = json.load(open(config_file))
        prediction_head = cls.subclasses[config["name"]](**config)
        model_file = cls._get_model_file(config_file=config_file)
        logger.info("Loading prediction head from {}".format(model_file))
        prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        return prediction_head

    def logits_to_loss(self, logits, labels):
        """
        Implement this function in your special Prediction Head.
        Should combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param labels: labels, can vary in shape and type, depending on task
        :type labels: object
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        """
        Implement this function in your special Prediction Head.
        Should combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def prepare_labels(self, **kwargs):
        """
        Some prediction heads need additional label conversion.
        E.g. NER needs word level labels turned into subword token level labels.

        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: labels in the right format
        :rtype: object
        """
        # TODO maybe just return **kwargs to not force people to implement this
        raise NotImplementedError()

    @classmethod
    def _get_model_file(cls, config_file):
        if "config.json" in config_file and "prediction_head" in config_file:
            head_num = int("".join([char for char in os.path.basename(config_file) if char.isdigit()]))
            model_file = os.path.join(os.path.dirname(config_file), f"prediction_head_{head_num}.bin")
        else:
            raise ValueError(f"This doesn't seem to be a proper prediction_head config file: '{config_file}'")
        return model_file

    def _set_name(self, name):
        self.task_name = name


class RegressionHead(PredictionHead):
    def __init__(
        self,
        layer_dims,
        task_name="regression",
        **kwargs,
    ):
        super(RegressionHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = 2
        self.ph_output_type = "per_sequence_continuous"
        self.model_type = "regression"
        self.loss_fct = MSELoss(reduction="none")
        self.task_name = task_name
        self.generate_config()

    def forward(self, x):
        logits = self.feed_forward(x)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        # Squeeze the logits to obtain a coherent output size
        label_ids = kwargs.get(self.label_tensor_name)
        return self.loss_fct(logits.squeeze(), label_ids.float())

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.cpu().numpy()
        #rescale predictions to actual label distribution
        preds = [x * self.label_list[1] + self.label_list[0] for x in preds]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        label_ids = [x * self.label_list[1] + self.label_list[0] for x in label_ids]
        return label_ids

    def formatted_preds(self, logits, samples, **kwargs):
        preds = self.logits_to_preds(logits)
        contexts = [sample.clear_text["text"] for sample in samples]

        assert len(preds) == len(contexts)

        res = {"task": "regression", "predictions": []}
        for pred, context in zip(preds, contexts):
            res["predictions"].append(
                {
                    "context": f"{context}",
                    "pred": pred[0]
                }
            )
        return res


class TextClassificationHead(PredictionHead):
    def __init__(
        self,
        layer_dims,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        task_name="text_classification",
        **kwargs,
    ):
        super(TextClassificationHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = task_name #used for connecting with the right output of the processor
        self.class_weights = class_weights

        if class_weights:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None

        self.loss_fct = CrossEntropyLoss(
            weight=balanced_weights,
            reduction=loss_reduction,
            ignore_index=loss_ignore_index,
        )

        self.generate_config()

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_probs(self, logits, return_class_probs, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_class_probs:
            probs = probs
        else:
            probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [self.label_list[int(x)] for x in pred_ids]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        labels = [self.label_list[int(x)] for x in label_ids]
        return labels

    def formatted_preds(self, logits, samples, return_class_probs=False, **kwargs):
        preds = self.logits_to_preds(logits)
        probs = self.logits_to_probs(logits, return_class_probs)
        contexts = [sample.clear_text["text"] for sample in samples]

        assert len(preds) == len(probs) == len(contexts)

        res = {"task": "text_classification", "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            if not return_class_probs:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            else:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": "class_probabilities",
                    "probability": prob,
                }

            res["predictions"].append(pred_dict)
        return res


class MultiLabelTextClassificationHead(PredictionHead):
    def __init__(
        self,
        layer_dims,
        class_weights=None,
        loss_reduction="none",
        task_name="text_classification",
        pred_threshold=0.5,
        **kwargs,
    ):
        super(MultiLabelTextClassificationHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "multilabel_text_classification"
        self.task_name = task_name #used for connecting with the right output of the processor
        self.class_weights = class_weights
        self.pred_threshold = pred_threshold

        if class_weights:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            #TODO must balanced weight really be a instance attribute?
            self.balanced_weights = nn.Parameter(
                torch.tensor(class_weights), requires_grad=False
            )
        else:
            self.balanced_weights = None

        self.loss_fct = BCEWithLogitsLoss(pos_weight=self.balanced_weights,
                                          reduction=loss_reduction)

        self.generate_config()

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name).to(dtype=torch.float)
        loss = self.loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1, self.num_labels))
        per_sample_loss = loss.mean(1)
        return per_sample_loss

    def logits_to_probs(self, logits, **kwargs):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        probs = self.logits_to_probs(logits)
        #TODO we could potentially move this to GPU to speed it up
        pred_ids = [np.where(row > self.pred_threshold)[0] for row in probs]
        preds = []
        for row in pred_ids:
            preds.append([self.label_list[int(x)] for x in row])
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        label_ids = [np.where(row == 1)[0] for row in label_ids]
        labels = []
        for row in label_ids:
            labels.append([self.label_list[int(x)] for x in row])
        return labels

    def formatted_preds(self, logits, samples, **kwargs):
        preds = self.logits_to_preds(logits)
        probs = self.logits_to_probs(logits)
        contexts = [sample.clear_text["text"] for sample in samples]

        assert len(preds) == len(probs) == len(contexts)

        res = {"task": "text_classification", "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            res["predictions"].append(
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
    def __init__(self, layer_dims, task_name="ner", **kwargs):
        super(TokenClassificationHead, self).__init__()

        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.ph_output_type = "per_token"
        self.model_type = "token_classification"
        self.task_name = task_name
        self.generate_config()

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(
        self, logits, initial_mask, padding_mask=None, **kwargs
    ):
        label_ids = kwargs.get(self.label_tensor_name)

        # Todo: should we be applying initial mask here? Loss is currently calculated even on non initial tokens
        active_loss = padding_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]

        loss = self.loss_fct(
            active_logits, active_labels
        )  # loss is a 1 dimemnsional (active) token loss
        return loss

    def logits_to_preds(self, logits, initial_mask, **kwargs):
        preds_word_all = []
        preds_tokens = torch.argmax(logits, dim=2)
        preds_token = preds_tokens.detach().cpu().numpy()
        # used to be: padding_mask = padding_mask.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()

        for idx, im in enumerate(initial_mask):
            preds_t = preds_token[idx]
            # Get labels and predictions for just the word initial tokens
            preds_word_id = self.initial_token_only(preds_t, initial_mask=im)
            preds_word = [self.label_list[pwi] for pwi in preds_word_id]
            preds_word_all.append(preds_word)
        return preds_word_all

    def logits_to_probs(self, logits, initial_mask, return_class_probs, **kwargs):
        # get per token probs
        softmax = torch.nn.Softmax(dim=2)
        token_probs = softmax(logits)
        if return_class_probs:
            token_probs = token_probs
        else:
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

    def prepare_labels(self, initial_mask, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        labels_all = []
        label_ids = label_ids.cpu().numpy()
        for label_ids_one_sample, initial_mask_one_sample in zip(
            label_ids, initial_mask
        ):
            label_ids = self.initial_token_only(
                label_ids_one_sample, initial_mask_one_sample
            )
            labels = [self.label_list[l] for l in label_ids]
            labels_all.append(labels)
        return labels_all

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret

    def formatted_preds(self, logits, initial_mask, samples, return_class_probs=False, **kwargs):
        preds = self.logits_to_preds(logits, initial_mask)
        probs = self.logits_to_probs(logits, initial_mask,return_class_probs)

        # align back with original input by getting the original word spans
        spans = []
        for sample, sample_preds in zip(samples, preds):
            word_spans = []
            span = None
            for token, offset, start_of_word in zip(
                sample.tokenized["tokens"],
                sample.tokenized["offsets"],
                sample.tokenized["start_of_word"],
            ):
                if start_of_word:
                    # previous word has ended unless it's the very first word
                    if span is not None:
                        word_spans.append(span)
                    span = {"start": offset, "end": offset + len(token)}
                else:
                    # expand the span to include the subword-token
                    span["end"] = offset + len(token.replace("##", ""))
            word_spans.append(span)
            spans.append(word_spans)

        assert len(preds) == len(probs) == len(spans)

        res = {"task": "ner", "predictions": []}
        for preds_seq, probs_seq, sample, spans_seq in zip(
            preds, probs, samples, spans
        ):
            tags, spans_seq = convert_iob_to_simple_tags(preds_seq, spans_seq)
            seq_res = []
            for tag, prob, span in zip(tags, probs_seq, spans_seq):
                context = sample.clear_text["text"][span["start"] : span["end"]]
                seq_res.append(
                    {
                        "start": span["start"],
                        "end": span["end"],
                        "context": f"{context}",
                        "label": f"{tag}",
                        "probability": prob,
                    }
                )
            res["predictions"].extend(seq_res)
        return res


class BertLMHead(PredictionHead):
    def __init__(self, hidden_size, vocab_size, hidden_act="gelu", task_name="lm", **kwargs):
        super(BertLMHead, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.num_labels = vocab_size  # vocab size
        # TODO Check if weight init needed!
        # self.apply(self.init_bert_weights)
        self.ph_output_type = "per_token"

        self.model_type = "language_modelling"
        self.task_name = task_name
        self.generate_config()

        # NN Layers
        # this is the "transform" module in the pytorch-transformers repo
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.transform_act_fn = ACT2FN[self.hidden_act]
        self.LayerNorm = BertLayerNorm(self.hidden_size, eps=1e-12)

        # this is the "decoder" in the pytorch-transformers repo
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size,
                                 vocab_size,
                                 bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    @classmethod
    def load(cls, pretrained_model_name_or_path, n_added_tokens=0):

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            if n_added_tokens != 0:
                #TODO resize prediction head decoder for custom vocab
                raise NotImplementedError("Custom vocab not yet supported for model loading from FARM files")

            config_file = os.path.exists(pretrained_model_name_or_path)
            model_file = cls._get_model_file(config_file)
            config = json.load(open(config_file))
            prediction_head = cls(**config)
            logger.info("Loading prediction head from {}".format(model_file))
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        else:
            # b) pytorch-transformers style
            # load weights from bert model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)

            # init empty head
            vocab_size = bert_with_lm.config.vocab_size + n_added_tokens

            head = cls(hidden_size=bert_with_lm.config.hidden_size,
                       vocab_size=vocab_size,
                       hidden_act=bert_with_lm.config.hidden_act)

            # load weights
            head.dense.load_state_dict(bert_with_lm.cls.predictions.transform.dense.state_dict())
            head.LayerNorm.load_state_dict(bert_with_lm.cls.predictions.transform.LayerNorm.state_dict())

            # Not loading weights of decoder here, since we later share weights with the embedding layer of LM
            #head.decoder.load_state_dict(bert_with_lm.cls.predictions.decoder.state_dict())

            if n_added_tokens == 0:
                bias_params = bert_with_lm.cls.predictions.bias
            else:
                # Custom vocab => larger vocab => larger dims of output layer in the LM head
                bias_params = torch.nn.Parameter(torch.cat([bert_with_lm.cls.predictions.bias,
                                                            torch.zeros(n_added_tokens)]))
            head.bias.data.copy_(bias_params)
            del bert_with_lm
            del bias_params

        return head

    def set_shared_weights(self, shared_embedding_weights):
        self.decoder.weight = shared_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        lm_logits = self.decoder(hidden_states) + self.bias
        return lm_logits

    def logits_to_loss(self, logits, **kwargs):
        lm_label_ids = kwargs.get(self.label_tensor_name)
        batch_size = lm_label_ids.shape[0]
        masked_lm_loss = self.loss_fct(
            logits.view(-1, self.num_labels), lm_label_ids.view(-1)
        )
        per_sample_loss = masked_lm_loss.view(-1, batch_size).mean(dim=0)
        return per_sample_loss

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        lm_label_ids = kwargs.get(self.label_tensor_name).cpu().numpy()
        lm_preds_ids = logits.argmax(2)
        # apply mask to get rid of predictions for non-masked tokens
        assert lm_preds_ids.shape == lm_label_ids.shape
        lm_preds_ids[lm_label_ids == -1] = -1
        lm_preds_ids = lm_preds_ids.tolist()
        preds = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for pred_ids_for_sequence in lm_preds_ids:
            preds.append(
                [self.label_list[int(x)] for x in pred_ids_for_sequence if int(x) != -1]
            )
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy().tolist()
        labels = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for ids_for_sequence in label_ids:
            labels.append([self.label_list[int(x)] for x in ids_for_sequence if int(x) != -1])
        return labels


class NextSentenceHead(TextClassificationHead):
    """
    Almost identical to a TextClassificationHead. Only difference: we can load the weights from
     a pretrained language model that was saved in the pytorch-transformers style (all in one model).
    """
    @classmethod
    def load(cls, pretrained_model_name_or_path):

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            config_file = os.path.exists(pretrained_model_name_or_path)
            # a) FARM style
            #TODO validate saving/loading after switching to processor.tasks
            model_file = cls._get_model_file(config_file)
            config = json.load(open(config_file))
            prediction_head = cls(**config)
            logger.info("Loading prediction head from {}".format(model_file))
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        else:
            # b) pytorch-transformers style
            # load weights from bert model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)

            # init empty head
            head = cls(layer_dims=[bert_with_lm.config.hidden_size, 2], loss_ignore_index=-1, task_name="nextsentence")

            # load weights
            head.feed_forward.feed_forward[0].load_state_dict(bert_with_lm.cls.seq_relationship.state_dict())
            del bert_with_lm

        return head

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


class QuestionAnsweringHead(PredictionHead):
    """
    A question answering head predicts the start and end of the answer on token level.
    """

    def __init__(self, layer_dims, task_name="question_answering", **kwargs):
        """
        :param layer_dims: dimensions of Feed Forward block, e.g. [768,2], for adjusting to BERT embedding. Output should be always 2
        :type layer_dims: List[Int]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        super(QuestionAnsweringHead, self).__init__()
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_token_squad"
        self.model_type = (
            "span_classification"
        )  # predicts start and end token of answer
        self.task_name = task_name
        self.generate_config()


    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Almost identical to a QuestionAnsweringHead. Only difference: we can load the weights from
         a pretrained language model that was saved in the pytorch-transformers style (all in one model).
        """

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            config_file = os.path.exists(pretrained_model_name_or_path)
            # a) FARM style
            model_file = cls._get_model_file(config_file)
            config = json.load(open(config_file))
            prediction_head = cls(**config)
            logger.info("Loading prediction head from {}".format(model_file))
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        else:
            # b) pytorch-transformers style
            # load weights from bert model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_qa = BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)

            # init empty head
            head = cls(layer_dims=[bert_qa.config.hidden_size, 2], loss_ignore_index=-1, task_name="question_answering")
            # load weights
            head.feed_forward.feed_forward[0].load_state_dict(bert_qa.qa_outputs.state_dict())
            del bert_qa

        return head


    def forward(self, X):
        """
        One forward pass through the prediction head model, starting with language model output on token level

        """
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        """
        Combine predictions and labels to a per sample loss.
        """
        # todo explain how we only use first answer for train
        # labels.shape =  [batch_size, n_max_answers, 2]. n_max_answers is by default 6 since this is the
        # most that occurs in the SQuAD dev set. The 2 in the final dimension corresponds to [start, end]
        start_position = labels[:, 0, 0]
        end_position = labels[:, 0, 1]

        # logits is of shape [batch_size, max_seq_len, 2]. Like above, the final dimension corresponds to [start, end]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Squeeze final singleton dimensions
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)

        ignored_index = start_logits.size(1)
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(reduction="none")
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        per_sample_loss = (start_loss + end_loss) / 2
        return per_sample_loss

    def logits_to_preds(self, logits, padding_mask, start_of_word, seq_2_start_t, max_answer_length=1000, **kwargs):
        """
        Get the predicted index of start and end token of the answer. Note that the output is at token level
        and not word level. Note also that these logits correspond to the tokens of a sample
        (i.e. special tokens, question tokens, passage_tokens)
        """

        # Will be populated with the top-n predictions of each sample in the batch
        # shape = batch_size x ~top_n
        # Note that ~top_n = n   if no_answer is     within the top_n predictions
        #           ~top_n = n+1 if no_answer is not within the top_n predictions
        all_top_n = []

        # logits is of shape [batch_size, max_seq_len, 2]. The final dimension corresponds to [start, end]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Calculate a few useful variables
        batch_size = start_logits.size()[0]
        max_seq_len = start_logits.shape[1] # target dim
        n_non_padding = torch.sum(padding_mask, dim=1)

        # get scores for all combinations of start and end logits => candidate answers
        start_matrix = start_logits.unsqueeze(2).expand(-1, -1, max_seq_len)
        end_matrix = end_logits.unsqueeze(1).expand(-1, max_seq_len, -1)
        start_end_matrix = start_matrix + end_matrix

        # Sort the candidate answers by their score. Sorting happens on the flattened matrix.
        # flat_sorted_indices.shape: (batch_size, max_seq_len^2, 1)
        flat_scores = start_end_matrix.view(batch_size, -1)
        flat_sorted_indices_2d = flat_scores.sort(descending=True)[1]
        flat_sorted_indices = flat_sorted_indices_2d.unsqueeze(2)

        # The returned indices are then converted back to the original dimensionality of the matrix.
        # sorted_candidates.shape : (batch_size, max_seq_len^2, 2)
        start_indices = flat_sorted_indices // max_seq_len
        end_indices = flat_sorted_indices % max_seq_len
        sorted_candidates = torch.cat((start_indices, end_indices), dim=2)

        # Get the n_best candidate answers for each sample that are valid (via some heuristic checks)
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(sorted_candidates[sample_idx], start_end_matrix[sample_idx],
                                                   n_non_padding[sample_idx], max_answer_length,
                                                   seq_2_start_t[sample_idx])
            all_top_n.append(sample_top_n)

        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix,
                           n_non_padding, max_answer_length, seq_2_start_t, n_best=5):
        """ Returns top candidate answers. Operates on a matrix of summed start and end logits. This matrix corresponds
        to a single sample (includes special tokens, question tokens, passage tokens). This method always returns a
        list of len n_best + 1 (it is comprised of the n_best positive answers along with the one no_answer)"""

        # Initialize some variables
        top_candidates = []
        n_candidates = sorted_candidates.shape[0]

        # Iterate over all candidates and break when we have all our n_best candidates
        for candidate_idx in range(n_candidates):
            if len(top_candidates) == n_best:
                break
            else:
                # Retrieve candidate's indices
                start_idx = sorted_candidates[candidate_idx, 0].item()
                end_idx = sorted_candidates[candidate_idx, 1].item()
                # Ignore no_answer scores which will be extracted later in this method
                if start_idx == 0 and end_idx == 0:
                    continue
                # Check that the candidate's indices are valid and save them if they are
                score = start_end_matrix[start_idx, end_idx].item()
                if self.valid_answer_idxs(start_idx, end_idx, n_non_padding, max_answer_length, seq_2_start_t):
                    # score = start_end_matrix[start_idx, end_idx].item()
                    top_candidates.append([start_idx, end_idx, score])

        no_answer_score = start_end_matrix[0, 0].item()
        top_candidates.append([0, 0, no_answer_score])

        return top_candidates

    @staticmethod
    def valid_answer_idxs(start_idx, end_idx, n_non_padding, max_answer_length, seq_2_start_t):
        """ Returns True if the supplied index span is a valid prediction. The indices being provided
        should be on sample/passage level (special tokens + question_tokens + passag_tokens)
        and not document level"""

        # This function can seriously slow down inferencing and eval
        # Continue if start or end label points to a padding token
        if start_idx < seq_2_start_t and start_idx != 0:
            return False
        if end_idx < seq_2_start_t and end_idx != 0:
            return False
        if start_idx >= n_non_padding:
            return False
        if end_idx >= n_non_padding:
            return False
        # Check if start comes after end
        if end_idx < start_idx:
            return False
        # If one of the two indices is 0, the other must also be 0
        if start_idx == 0 and end_idx != 0:
            return False
        if start_idx != 0 and end_idx == 0:
            return False

        length = end_idx - start_idx + 1
        if length > max_answer_length:
            return False
        return True

    def formatted_preds(self, logits, baskets, rest_api_schema=False):
        """ Takes a list of logits, each corresponding to one sample, and converts them into document level predictions.
        Leverages information in the SampleBaskets. Assumes that we are being passed logits from ALL samples in the one
        SampleBasket i.e. all passages of a document. """

        # Unpack some useful variables
        # passage_start_t is the token index of the passage relative to the document (usually a multiple of doc_stride)
        # seq_2_start_t is the token index of the first token in passage relative to the input sequence (i.e. number of
        # special tokens and question tokens that come before the passage tokens)
        samples = [s for b in baskets for s in b.samples]
        ids = [s.id.split("-") for s in samples]
        passage_start_t = [s.features[0]["passage_start_t"] for s in samples]
        seq_2_start_t = [s.features[0]["seq_2_start_t"] for s in samples]

        # Prepare tensors
        logits = torch.stack(logits)
        padding_mask = torch.tensor([s.features[0]["padding_mask"] for s in samples], dtype=torch.long)
        start_of_word = torch.tensor([s.features[0]["start_of_word"] for s in samples], dtype=torch.long)

        # Return n + 1 predictions per passage / sample
        preds_p = self.logits_to_preds(logits, padding_mask, start_of_word, seq_2_start_t)

        # Aggregate passage level predictions to create document level predictions.
        # This method assumes that all passages of each document are contained in preds_p
        # i.e. that there are no incomplete documents. The output of this step
        # are prediction spans
        preds_d = self.aggregate_preds(preds_p, passage_start_t, ids, seq_2_start_t)
        assert len(preds_d) == len(baskets)

        # Separate top_preds list from the no_ans_gap float
        top_preds, no_ans_gaps = zip(*preds_d)

        # Takes document level prediction spans and returns string predictions
        formatted = self.stringify(top_preds, baskets)

        if rest_api_schema:
            formatted = self.to_rest_api_schema(formatted, no_ans_gaps, baskets)

        return formatted

    def stringify(self, top_preds, baskets):
        """ Turn prediction spans into strings """
        ret = []

        # Iterate over each set of document level prediction
        for pred_d, basket in zip(top_preds, baskets):
            curr_dict = {}
            # Unpack document offsets, clear text and squad_id
            token_offsets = basket.raw["document_offsets"]
            clear_text = basket.raw["document_text"]
            squad_id = basket.raw["squad_id"]

            # Iterate over each prediction on the one document
            full_preds = []
            for start_t, end_t, score in pred_d:
                pred_str, _, _ = self.span_to_string(start_t, end_t, token_offsets, clear_text)
                full_preds.append([pred_str, start_t, end_t, score])
            curr_dict["id"] = squad_id
            curr_dict["preds"] = full_preds
            ret.append(curr_dict)
        return ret


    def to_rest_api_schema(self, formatted_preds, no_ans_gaps, baskets):
        ret = []
        ids = [fp["id"] for fp in formatted_preds]
        preds = [fp["preds"] for fp in formatted_preds]

        for preds, id, no_ans_gap, basket in zip(preds, ids, no_ans_gaps, baskets):
            question = basket.raw["question_text"]
            answers = self.answer_for_api(preds, basket)
            curr = {
                "task": "qa",
                "predictions": [
                    {
                        "question": question,
                        "question_id": id,
                        "ground_truth": None,
                        "answers": answers,
                        "no_ans_gap": no_ans_gap
                    }
                ],
            }
            ret.append(curr)
        return ret

    def answer_for_api(self, top_preds, basket):
        ret = []
        token_offsets = basket.raw["document_offsets"]
        clear_text = basket.raw["document_text"]

        # iterate over the top_n predictions of the one document
        for string, start_t, end_t, score in top_preds:

            _, ans_start_ch, ans_end_ch = self.span_to_string(start_t, end_t, token_offsets, clear_text)
            context_string, context_start_ch, context_end_ch = self.create_context(ans_start_ch, ans_end_ch, clear_text)
            curr = {"score": score,
                    "probability": -1,
                    "answer": string,
                    "offset_answer_start": ans_start_ch,
                    "offset_answer_end": ans_end_ch,
                    "context": context_string,
                    "offset_context_start": context_start_ch,
                    "offset_context_end": context_end_ch,
                    "document_id": None}
            ret.append(curr)
        return ret

    def create_context(self, ans_start_ch, ans_end_ch, clear_text, window_size_ch=100):
        if ans_start_ch == 0 and ans_end_ch == 0:
            context_start_ch = 0
            context_end_ch = 0
        else:
            len_text = len(clear_text)
            midpoint = int((ans_end_ch - ans_start_ch) / 2)
            half_window = int(window_size_ch / 2)
            context_start_ch = midpoint - half_window
            context_end_ch = midpoint + half_window
            overhang_start = -context_start_ch
            overhang_end = len_text - context_end_ch
            context_start_ch -= overhang_end
            context_start_ch = max(0, context_start_ch)
            context_end_ch += overhang_start
            context_end_ch = min(len_text, context_end_ch)
        context_string = clear_text[context_start_ch: context_end_ch]
        return context_string, context_start_ch, context_end_ch

    @staticmethod
    def span_to_string(start_t, end_t, token_offsets, clear_text):

        # If it is a no_answer prediction
        if start_t == -1 and end_t == -1:
            return "", 0, 0

        n_tokens = len(token_offsets)

        # We do this to point to the beginning of the first token after the span instead of
        # the beginning of the last token in the span
        end_t += 1

        # Predictions sometimes land on the very final special token of the passage. But there are no
        # special tokens on the document level. We will just interpret this as a span that stretches
        # to the end of the document
        end_t = min(end_t, n_tokens)

        start_ch = token_offsets[start_t]
        # i.e. pointing at the END of the last token
        if end_t == n_tokens:
            end_ch = len(clear_text)
        else:
            end_ch = token_offsets[end_t]
        return clear_text[start_ch: end_ch].strip(), start_ch, end_ch

    def has_no_answer_idxs(self, sample_top_n):
        for start, end, score in sample_top_n:
            if start == 0 and end == 0:
                return True
        return False

    def aggregate_preds(self, preds, passage_start_t, ids, seq_2_start_t=None, labels=None):
        """ Aggregate passage level predictions to create document level predictions.
        This method assumes that all passages of each document are contained in preds
        i.e. that there are no incomplete documents. The output of this step
        are prediction spans. No answer is represented by a (-1, -1) span on the document level """

        # Initialize some variables
        n_samples = len(preds)
        all_basket_preds = {}
        all_basket_labels = {}

        # Iterate over the preds of each sample
        for sample_idx in range(n_samples):
            # Aggregation of sample predictions is done using these ids
            # See SquadProcessor.convert_squad_id for why there are two parts
            squad_id_1, squad_id_2, _ = ids[sample_idx]
            basket_id = f"{squad_id_1}-{squad_id_2}"

            # curr_passage_start_t is the token offset of the current passage
            # It will always be a multiple of doc_stride
            curr_passage_start_t = passage_start_t[sample_idx]

            # This is to account for the fact that all model input sequences start with some special tokens
            # and also the question tokens before passage tokens.
            if seq_2_start_t:
                cur_seq_2_start_t = seq_2_start_t[sample_idx]
                curr_passage_start_t -= cur_seq_2_start_t

            # Converts the passage level predictions+labels to document level predictions+labels. Note
            # that on the passage level a no answer is (0,0) but at document level it is (-1,-1) since (0,0)
            # would refer to the first token of the document
            pred_d = self.pred_to_doc_idxs(preds[sample_idx], curr_passage_start_t)
            if labels:
                label_d = self.label_to_doc_idxs(labels[sample_idx], curr_passage_start_t)

            # Initialize the basket_id as a key in the all_basket_preds and all_basket_labels dictionaries
            if basket_id not in all_basket_preds:
                all_basket_preds[basket_id] = []
                all_basket_labels[basket_id] = []

            # Add predictions and labels to dictionary grouped by their basket_ids
            all_basket_preds[basket_id].append(pred_d)
            if labels:
                all_basket_labels[basket_id].append(label_d)

        # Pick n-best predictions and remove repeated labels
        all_basket_preds = {k: self.reduce_preds(v) for k, v in all_basket_preds.items()}
        if labels:
            all_basket_labels = {k: self.reduce_labels(v) for k, v in all_basket_labels.items()}

        # Return aggregated predictions in order as a list of lists
        keys = [k for k in all_basket_preds]
        aggregated_preds = [all_basket_preds[k] for k in keys]
        if labels:
            labels = [all_basket_labels[k] for k in keys]
            return aggregated_preds, labels
        else:
            return aggregated_preds

    @staticmethod
    def reduce_labels(labels):
        """ Removes repeat answers. Represents a no answer label as (-1,-1)"""
        positive_answers = [(start, end) for x in labels for start, end in x if not (start == -1 and end == -1)]
        if not positive_answers:
            return [(-1, -1)]
        else:
            return list(set(positive_answers))

    def reduce_preds(self, preds, n_best=5):
        """ This function contains the logic for choosing the best answers from each passage. In the end, it
        returns the n_best predictions on the document level. """

        # Initialize some variables
        document_no_answer = True
        passage_no_answer = []
        passage_best_score = []
        no_answer_scores = []

        # Iterate over the top predictions for each sample
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred[2]
            no_answer_score = self.get_no_answer_score(sample_preds)
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            passage_best_score.append(best_pred_score)

        # If a positive prediction is higher than the no_answer score in one of the passages then the top
        # document prediction should be a positive answer
        if False in passage_no_answer:
            document_no_answer = False

        # Get all predictions in flattened list and sort by score
        pos_answers_flat = [(start, end, score)
                            for passage_preds in preds
                            for start, end, score in passage_preds
                            if not (start == -1 and end == -1)]
        pos_answers_sorted = sorted(pos_answers_flat, key=lambda x: x[2], reverse=True)
        pos_answers_reduced = pos_answers_sorted[:n_best]
        no_answer_pred = [-1, -1, max(no_answer_scores)]

        # TODO this is how big the no_answer threshold needs to be to change a no_answer to a pos answer
        #  (or vice versa). This can in future be used to train the threshold value
        no_ans_gap = max([nas - pbs for nas, pbs in zip(no_answer_scores, passage_best_score)])

        if document_no_answer:
            n_preds = [no_answer_pred] + pos_answers_reduced[:-1]
        else:
            n_preds = pos_answers_reduced
        return n_preds, no_ans_gap

    ## THIS IS A SIMPLER IMPLEMENTATION OF PICKING BEST ANSWERS FOR A DOCUMENT. MATCHES THE HUGGINGFACE METHOD
    # @staticmethod
    # def reduce_preds(preds, n_best=5):
    #     pos_answers = [[(start, end, score) for start, end, score in x if not (start == -1 and end == -1)] for x in preds]
    #     pos_answer_flat = [x for y in pos_answers for x in y]
    #     pos_answers_sorted = sorted(pos_answer_flat, key=lambda z: z[2], reverse=True)
    #     pos_answers_filtered = pos_answers_sorted[:n_best]
    #     top_pos_answer_score = pos_answers_filtered[0][2]
    #
    #     no_answer = [(start, end, score) for x in preds for start, end, score in x if (start == -1 and end == -1)]
    #     no_answer_sorted = sorted(no_answer, key=lambda z: z[2], reverse=True)
    #     no_answers_min = no_answer_sorted[-1]
    #     _, _, no_answer_min_score = no_answers_min
    #
    #     # no answer logic
    #     threshold = 0.
    #     if no_answer_min_score + threshold > top_pos_answer_score:
    #         return [no_answers_min] + pos_answers_filtered
    #     else:
    #         return pos_answers_filtered + [no_answers_min]

    @staticmethod
    def get_no_answer_score(preds):
        for start, end, score in preds:
            if start == -1 and end == -1:
                return score
        raise Exception

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t):
        """ Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) span but will instead be represented by (-1, -1)"""
        new_pred = []
        for start, end, score in pred:
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                assert start >= 0
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                assert start >= 0
            new_pred.append([start, end, score])
        return new_pred

    @staticmethod
    def label_to_doc_idxs(label, passage_start_t):
        """ Converts the passage level labels to document level labels. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) span but will instead be represented by (-1, -1)"""
        new_label = []
        for start, end in label:
            # If there is a valid label
            if start > 0 or end > 0:
                new_label.append((start + passage_start_t, end + passage_start_t))
            # If the label is a no answer, we represent this as a (-1, -1) span
            # since there is no CLS token on the document level
            if start == 0 and end == 0:
                new_label.append((-1, -1))
        return new_label

    def prepare_labels(self, labels, start_of_word, **kwargs):
        return labels
