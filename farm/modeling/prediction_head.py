import itertools
import json
import logging
import os
import numpy as np
import pandas as pd
from scipy.special import expit, softmax

import torch
from transformers.modeling_bert import BertForPreTraining, BertLayerNorm, ACT2FN, BertForQuestionAnswering

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from farm.data_handler.utils import is_json
from farm.utils import convert_iob_to_simple_tags

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
    def load(cls, pretrained_model_name_or_path):

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
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)

            # init empty head
            head = cls(hidden_size=bert_with_lm.config.hidden_size,
                       vocab_size=bert_with_lm.config.vocab_size,
                       hidden_act=bert_with_lm.config.hidden_act)

            # load weights
            head.dense.load_state_dict(bert_with_lm.cls.predictions.transform.dense.state_dict())
            head.LayerNorm.load_state_dict(bert_with_lm.cls.predictions.transform.LayerNorm.state_dict())

            head.decoder.load_state_dict(bert_with_lm.cls.predictions.decoder.state_dict())
            head.bias.data.copy_(bert_with_lm.cls.predictions.bias)
            del bert_with_lm

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

    def __init__(self,
                 layer_dims,
                 task_name="question_answering",
                 no_answer_shift=0,
                 top_n_predictions=3,
                 context_size=100,
                 **kwargs):
        """
        :param layer_dims: dimensions of Feed Forward block, e.g. [768,2], for adjusting to BERT embedding. Output should be always 2
        :type layer_dims: List[Int]
        :param task_name: Name of task
        :type task_name: str
        :param no_answer_shift: How much we want to weight giving no answer compared to text answer
                                We actually compare in logit space (sum of both start and end logit), so negative
                                values result in less no answer predictions and vice versa. normal range = [-5,+5]
        :type no_answer_shift: int
        :param top_n_predictions: When we split a document into multiple passages we can return top n passage answers
        :type top_n_predictions: int
        :param context_size: When we format predictions back to string space we also return surrounding context
                             of size context_size
        :type context_size: int
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
        self.no_answer_shift = no_answer_shift # how much we want to upweight no answer logit scores compared to text answer ones
        self.top_n_predictions = top_n_predictions #for how many passages we want to get predictions
        self.context_size = context_size
        self.max_ans_len = 1000 # disabling max ans len. Impact on squad performance seems minor
        # each answer is returned with surrounding context. In # characters surrounding the answer
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

        :param X: Output of language model, of shape [batch_size, seq_length, LM_embedding_dim]
        :type X: torch.tensor
        :return: (start_logits, end_logits), logits for the start and end of answer
        :rtype: tuple[torch.tensor,torch.tensor]
        """
        logits = self.feed_forward(X)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return (start_logits, end_logits)

    def logits_to_loss(self, logits, start_position, end_position, **kwargs):
        """
        Combine predictions and labels to a per sample loss.

        :param logits: (start_logits, end_logits), logits for the start and end of answer
        :type logits: tuple[torch.tensor,torch.tensor]
        :param start_position: tensor with indices of START positions per sample
        :type start_position: torch.tensor
        :param end_position: tensor with indices of END positions per sample
        :type end_position: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: per_sample_loss: Per sample loss : )
        :rtype: torch.tensor
        """
        (start_logits, end_logits) = logits

        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)
        # sometimes the start/end positions (the labels read from file) are outside our model predictions, we ignore these terms
        # TODO check if ignored_index is needed. We are checking for start + end validity during construction of samples
        ignored_index = start_logits.size(1)
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        per_sample_loss = (start_loss + end_loss) / 2
        return per_sample_loss

    def logits_to_preds(self, logits, **kwargs):
        """
        Get the predicted index of start and end token of the answer.

        :param logits: (start_logits, end_logits), logits for the start and end of answer
        :type logits: tuple[torch.tensor,torch.tensor]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: (start_idx, end_idx), start and end indices for all samples in batch
        :rtype: (torch.tensor,torch.tensor)
        """

        # cast data into useful types/shapes
        (start_logits, end_logits) = logits
        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()
        num_per_batch = start_logits.shape[0]
        segment_ids = kwargs['segment_ids'].data.cpu().numpy()
        sample_ids = kwargs["sample_id"].cpu().numpy()
        passage_shifts = kwargs["passage_shift"].cpu().numpy()

        question_shifts = np.argmax(segment_ids > 0,axis=1)


        no_answer_sum = start_logits[:,0] + end_logits[:,0]
        best_answer_sum = np.zeros(num_per_batch)
        # check if start or end point to the context. Context starts at segment id == 1  (question comes before at segment ids == 0)
        context_start = np.argmax(segment_ids,axis=1)
        context_end = segment_ids.shape[1] - np.argmax(segment_ids[::-1],axis=1)
        start_proposals = self._get_best_textanswer_indices(start_logits, 3)
        end_proposals = self._get_best_textanswer_indices(end_logits, 3)
        best_indices = np.zeros((num_per_batch,2),dtype=int)
        for i_batch in range(num_per_batch):
            # for each sample create mesh of possible start + end combinations and their score as sum of logits
            mesh_idx = np.meshgrid(start_proposals[i_batch,:],end_proposals[i_batch])
            start_comb = mesh_idx[0].flatten()
            end_comb = mesh_idx[1].flatten()
            scores = start_logits[i_batch,start_comb] + end_logits[i_batch,end_comb]
            #iterate over combinations and eliminate impossible ones
            for idx in np.argsort(scores)[::-1]:
                start = start_comb[idx]
                end = end_comb[idx]
                if start < context_start[i_batch]:
                    continue
                if end > context_end[i_batch]:
                    continue
                if start > end:
                    continue
                if(end - start > self.max_ans_len):
                    continue
                # maybe need check weather start/end idx refers to start of word and not to a ##... continuation

                best_indices[i_batch,0] = start
                best_indices[i_batch,1] = end
                best_answer_sum[i_batch] = scores[idx]
                break # since we take most likely predictions first, we stop when finding a valid prediction

        # For each predicted text answer, we want to check weather this question could also be unanswerable with
        # the given passage
        # we need to compare the text answer logits with no-answer logits (at position 0)
        idx_no_answer = no_answer_sum + self.no_answer_shift >= best_answer_sum
        best_indices[idx_no_answer,:] = 0
        best_answer_sum[idx_no_answer] = no_answer_sum[idx_no_answer]

        # #probabilities computed through softmaxing logits. Currently unused in downstream code.
        # start_probs = softmax(start_logits, axis=1)
        # end_probs = softmax(end_logits, axis=1)
        # probabilities = (start_probs[range(best_indices.shape[0]), np.squeeze(best_indices[:, 0])] +
        #                  end_probs[range(best_indices.shape[0]), np.squeeze(best_indices[:, 1])]) / 2

        return (best_indices[:, 0], best_indices[:, 1], best_answer_sum, sample_ids, question_shifts, passage_shifts)

    def prepare_labels(self, start_position, end_position, **kwargs):
        """
        We want to pack labels into a tuple, to be compliant with later functions

        :param start_position: indices of answer start positions (in token space)
        :type start_position: torch.tensor
        :param end_position: indices of answer end positions (in token space)
        :type end_position: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: tuplefied sample_id with corresponding positions
        :rtype: tuple(torch.tensor, torch.tensor,torch.tensor)
        """
        return (kwargs["sample_id"], start_position, end_position)

    def formatted_preds(self, logits, preds, samples):
        """
        Format predictions into actual answer strings (substrings of context). Used for Inference!

        :param logits: palceholder to comply with LanguageModel.formatted_preds
        :type logits: None
        :param preds: predictions for each passage, coming from logits_to_preds()
                      contains start_idxs, end_idxs, logit_sums, sample_ids, question_shifts, passage_shifts, probabilities
        :type preds: tuple( 7 numpy arrays )
        :param samples: converted samples, to get a hook onto the actual text
        :type samples: List[FARM.data_handler.samples.Sample]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: Answers to the (ultimate) questions
        :rtype: list(str)
        """

        all_preds_passage_aggregated = self._aggregate_preds(preds=preds)

        result = {}
        result["task"] = "qa"
        all_preds = []
        sample_id_to_index = dict([(sample.id,i) for i,sample in enumerate(samples)])
        for current_pred in all_preds_passage_aggregated:
            sampleid_i = current_pred[0, 3]
            try:
                current_sample = samples[sample_id_to_index[sampleid_i]]
            except Exception as e:
                current_sample = None
                logger.warning(f"Sample id: {sampleid_i} could not be loaded. Error: {e} ")
            if current_sample is not None:
                passage_predictions = []
                for i in range(current_pred.shape[0]):
                    passage_pred = {}
                    if self.top_n_predictions > 1:
                        passage_pred["prediction_rank"] = i
                    s_i = current_pred[i,0]
                    e_i = current_pred[i,1]
                    logit_sum_i = current_pred[i,2]
                    question_shift_i = current_pred[i,4]
                    passage_shift_i = current_pred[i,5]
                    passage_pred["score"] = logit_sum_i
                    passage_pred["probability"] = -1 # TODO add probabilities that make sense : )
                    try:
                        #default to returning no answer
                        start = 0
                        end = 0
                        context_start = 0
                        answer = ""
                        context = ""
                        if(s_i + e_i > 0):
                            current_start = int(s_i + passage_shift_i - question_shift_i)
                            current_end = int(e_i + passage_shift_i - question_shift_i) + 1
                            start = current_sample.tokenized["offsets"][current_start]
                            end = current_sample.tokenized["offsets"][current_end]
                            # we want the answer in original string space (containing newline, tab or multiple
                            # whitespace. So we need to join doc tokens and work with character offsets
                            temptext = " ".join(current_sample.clear_text["doc_tokens"])
                            answer = temptext[start:end]
                            answer = answer.strip()
                            # sometimes we strip trailing whitespaces, so we need to adjust end
                            end = start + len(answer)
                            context_start = int(np.clip((start-self.context_size),a_min=0,a_max=None))
                            context_end = int(np.clip(end +self.context_size,a_max=len(temptext),a_min=None))
                            context = temptext[context_start:context_end]
                    except IndexError as e:
                        logger.info(e)
                    passage_pred["answer"] = answer
                    passage_pred["offset_start"] = start
                    passage_pred["offset_end"] = end
                    passage_pred["context"] = context
                    passage_pred["offset_context_start"] = start - context_start
                    passage_pred["offset_context_end"] = end - context_start
                    passage_pred["document_id"] = current_sample.clear_text.get("document_id", None)
                    passage_predictions.append(passage_pred)

                pred = {}
                pred["question"] = current_sample.clear_text.question_text
                pred["question_id"] = current_sample.clear_text.get("qas_id", None)
                pred["ground_truth"] = current_sample.clear_text.get("orig_answer_text", None)
                pred["answers"] = passage_predictions
                all_preds.append(pred)

        result["predictions"] = all_preds
        return result

    def _aggregate_preds(self, preds):
        def create_answeridx_string(r):
            start = r["pred_start"] + r["passage_shift"]
            end = r["pred_end"] + r["passage_shift"]
            return f"{start}-{end}"

        data = {}
        data["pred_start"] = np.concatenate([x[0] for x in preds])
        data["pred_end"] = np.concatenate([x[1] for x in preds])
        data["logit_sum"] = np.concatenate([x[2] for x in preds])
        data["sample_id"] = np.concatenate([x[3] for x in preds])
        data["question_shift"] = np.concatenate([x[4] for x in preds])
        data["passage_shift"] = np.concatenate([x[5] for x in preds])
        df = pd.DataFrame(data=data)
        df.loc[:,"answer_indices"] = df.apply(lambda row: create_answeridx_string(row), axis=1)

        # we sometimes have multiple predictions for one sample (= paragraph question pair)
        # because we split the paragraph into smaller passages
        # we group all predictions by sample_id
        unique_sample_ids = df.sample_id.unique()
        max_per_sample = []
        for uid in unique_sample_ids:
            group = df.loc[df.sample_id == uid, :]
            idx_text_answers = (group.pred_start + group.pred_end) > 0
            # if we have a text answer in the group we want to discard all no text passages
            # Reasoning: consider how data is constructed from labels: If we split a document into 5 passages and
            # only passage no.3 has the text answer, all remaining passages are labeled as no answer.
            # We want the answer for passage no.3 without regarding the models output for the other passages.
            if np.sum(idx_text_answers) > 0:
                group = group.loc[idx_text_answers,:]
            else:
                logger.info(f"No textual prediction found in doc: {uid}")
            if self.top_n_predictions == 1:
                max_pred = group.loc[group.logit_sum == np.max(group.logit_sum),:]
                if (max_pred.shape[0] > 1):
                    max_pred = max_pred.iloc[0, :]
                    logger.info(f"Multiple predictions have the exact same probability of occuring: \n{max_pred.head()}")
            else:
                assert isinstance(self.top_n_predictions, int)
                sorted_group = group.sort_values(by="logit_sum",ascending=False)
                filtered_group = sorted_group.drop_duplicates(keep="first", subset="answer_indices")
                max_pred = filtered_group.iloc[:self.top_n_predictions,:]
            max_per_sample.append(max_pred.values)
        return max_per_sample


    def _get_best_textanswer_indices(self, logits, n_best_size):
        """Get the n-best logits from a numpy array without considering the zero index.
        zero index corresponds to no answer, which we deal with separately"""
        logits_without_zero = logits[:,1:]
        idx_without_zero = np.argsort(logits_without_zero,axis=1)[:,-n_best_size:]
        idx = idx_without_zero + 1
        return idx
