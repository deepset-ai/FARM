import json
import logging
import os
import numpy as np

from pathlib import Path
from transformers.models.bert.modeling_bert import BertForPreTraining, ACT2FN
from transformers import AutoModelForQuestionAnswering, AutoModelForTokenClassification, AutoModelForSequenceClassification
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, NLLLoss
from farm.data_handler.utils import is_json
from farm.utils import convert_iob_to_simple_tags, try_get, all_gather_list
from farm.modeling.predictions import QACandidate, QAPred

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


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
        :type save_dir: str or Path
        :param head_num: Which head to save
        :type head_num: int
        """
        # updating config in case the parameters have been changed
        self.generate_config()
        output_config_file = Path(save_dir) / f"prediction_head_{head_num}_config.json"
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def save(self, save_dir, head_num=0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :type save_dir: str or Path
        :param head_num: which head to save
        :type head_num: int
        """
        output_model_file = Path(save_dir) / f"prediction_head_{head_num}.bin"
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if is_json(value) and key[0] != "_":
                config[key] = value
            if self.task_name == "text_similarity" and key == "similarity_function":
                config['similarity_function'] = value
        config["name"] = self.__class__.__name__
        config.pop("config", None)
        self.config = config

    @classmethod
    def load(cls, config_file, strict=True, load_weights=True):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :type config_file: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        config = json.load(open(config_file))
        prediction_head = cls.subclasses[config["name"]](**config)
        if load_weights:
            model_file = cls._get_model_file(config_file=config_file)
            logger.info("Loading prediction head from {}".format(model_file))
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")), strict=strict)
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

    def resize_input(self, input_dim):
        """ This function compares the output dimensionality of the language model against the input dimensionality
        of the prediction head. If there is a mismatch, the prediction head will be resized to fit."""
        if "feed_forward" not in dir(self):
            return
        else:
            old_dims = self.feed_forward.layer_dims
            if input_dim == old_dims[0]:
                return
            new_dims = [input_dim] + old_dims[1:]
            logger.info(f"Resizing input dimensions of {type(self).__name__} ({self.task_name}) "
                  f"from {old_dims} to {new_dims} to match language model")
            self.feed_forward = FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file):
        if "config.json" in str(config_file) and "prediction_head" in str(config_file):
            head_num = int("".join([char for char in os.path.basename(config_file) if char.isdigit()]))
            model_file = Path(os.path.dirname(config_file)) / f"prediction_head_{head_num}.bin"
        else:
            raise ValueError(f"This doesn't seem to be a proper prediction_head config file: '{config_file}'")
        return model_file

    def _set_name(self, name):
        self.task_name = name


class RegressionHead(PredictionHead):
    def __init__(
        self,
        layer_dims=[768,1],
        task_name="regression",
        **kwargs,
    ):
        super(RegressionHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        # num_labels is being set to 2 since it is being hijacked to store the scaling factor and the mean
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
        label_ids = kwargs.get(self.label_tensor_name)
        return self.loss_fct(logits, label_ids.float())

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
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        task_name="text_classification",
        **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_ignore_index:
        :param loss_reduction:
        :param task_name:
        :param kwargs:
        """
        super(TextClassificationHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = task_name #used for connecting with the right output of the processor

        if type(class_weights) is np.ndarray and class_weights.ndim != 1:
            raise ValueError("When you pass `class_weights` as `np.ndarray` it must have 1 dimension! "
                             "You provided {} dimensions.".format(class_weights.ndim))

        self.class_weights = class_weights

        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None

        self.loss_fct = CrossEntropyLoss(
            weight=balanced_weights,
            reduction=loss_reduction,
            ignore_index=loss_ignore_index,
        )

        # add label list
        if "label_list" in kwargs:
            self.label_list = kwargs["label_list"]

        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public name:
                                              - deepset/bert-base-german-cased-hatespeech-GermEval18Coarse

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            head = super(TextClassificationHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, revision=revision)
            # init empty head
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.id2label)])
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            # add label list
            head.label_list = list(full_model.config.id2label.values())
            del full_model

        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids
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
        # This is the standard doc classification case
        try:
            labels = [self.label_list[int(x)] for x in label_ids]
        # This case is triggered in Natural Questions where each example can have multiple labels
        except TypeError:
            labels = [self.label_list[int(x[0])] for x in label_ids]
        return labels

    def formatted_preds(self, logits=None, preds=None, samples=None, return_class_probs=False, **kwargs):
        """ Like QuestionAnsweringHead.formatted_preds(), this fn can operate on either logits or preds. This
        is needed since at inference, the order of operations is very different depending on whether we are performing
        aggregation or not (compare Inferencer._get_predictions() vs Inferencer._get_predictions_and_aggregate())"""

        assert (logits is not None) or (preds is not None)

        # When this method is used along side a QAHead at inference (e.g. Natural Questions), preds is the input and
        # there is currently no good way of generating probs
        if logits is not None:
            preds = self.logits_to_preds(logits)
            probs = self.logits_to_probs(logits, return_class_probs)
        else:
            probs = [None] * len(preds)

        # TODO this block has to do with the difference in Basket and Sample structure between SQuAD and NQ
        try:
            contexts = [sample.clear_text["text"] for sample in samples]
        # This case covers Natural Questions where the sample is in a QA style
        except KeyError:
            contexts = [sample.clear_text["question_text"] + " | " + sample.clear_text["passage_text"] for sample in samples]

        contexts_b = [sample.clear_text["text_b"] for sample in samples if "text_b" in  sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ["|".join([a, b]) for a,b in zip(contexts, contexts_b)]

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
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_reduction="none",
        task_name="text_classification",
        pred_threshold=0.5,
        **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_reduction:
        :param task_name:
        :param pred_threshold:
        :param kwargs:
        """
        super(MultiLabelTextClassificationHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.ph_output_type = "per_sequence"
        self.model_type = "multilabel_text_classification"
        self.task_name = task_name #used for connecting with the right output of the processor
        self.class_weights = class_weights
        self.pred_threshold = pred_threshold

        if class_weights is not None:
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
    def __init__(self,
                 layer_dims=None,
                 num_labels=None,
                 task_name="ner",
                 **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param task_name:
        :param kwargs:
        """
        super(TokenClassificationHead, self).__init__()
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.ph_output_type = "per_token"
        self.model_type = "token_classification"
        self.task_name = task_name
        if "label_list" in kwargs:
            self.label_list = kwargs["label_list"]
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased-finetuned-conll03-english)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased-finetuned-conll03-english

                                              See https://huggingface.co/models for full list

        """

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            head = super(TokenClassificationHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path, revision=revision)
            # init empty head
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.label2id)])
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            # add label list
            head.label_list = list(full_model.config.id2label.values())
            head.generate_config()
            del full_model
        return head


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
        spans = [s.tokenized["word_spans"] for s in samples]
        res = {"task": "ner", "predictions": []}
        for preds_seq, probs_seq, sample, spans_seq in zip(
            preds, probs, samples, spans
        ):
            tags, spans_seq = convert_iob_to_simple_tags(preds_seq, spans_seq)
            seq_res = []
            # TODO: Though we filter out tags and spans for non-entity words,
            # TODO: we do not yet filter out probs of non-entity words. This needs to be implemented still
            for tag, prob, span in zip(tags, probs_seq, spans_seq):
                context = sample.clear_text["text"][span[0]: span[1]]
                seq_res.append(
                    {
                        "start": span[0],
                        "end": span[1],
                        "context": f"{context}",
                        "label": f"{tag}",
                        "probability": np.float32(0.0),
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
        # Adding layer_dims (required for conversion to transformers)
        self.layer_dims = [hidden_size, vocab_size]
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
    def load(cls, pretrained_model_name_or_path, revision=None, n_added_tokens=0):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            if n_added_tokens != 0:
                #TODO resize prediction head decoder for custom vocab
                raise NotImplementedError("Custom vocab not yet supported for model loading from FARM files")

            head = super(BertLMHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) pytorch-transformers style
            # load weights from bert model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path, revision=revision)

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
        lm_label_ids = kwargs.get(self.label_tensor_name).cpu().numpy()
        lm_preds_ids = logits.argmax(2).cpu().numpy()
        # apply mask to get rid of predictions for non-masked tokens
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
     a pretrained language model that was saved in the Transformers style (all in one model).
    """
    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list

        """
        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            head = super(NextSentenceHead, cls).load(pretrained_model_name_or_path)
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
        self.layer_dims = layer_dims
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

    def __init__(self, layer_dims=[768,2],
                 task_name="question_answering",
                 no_ans_boost=0.0,
                 context_window_size=100,
                 n_best=5,
                 n_best_per_sample=1,
                 duplicate_filtering=-1,
                 **kwargs):
        """
        :param layer_dims: dimensions of Feed Forward block, e.g. [768,2], for adjusting to BERT embedding. Output should be always 2
        :type layer_dims: List[Int]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
                             The higher the value, the more likely a "no answer possible given the input text" is returned by the model
        :type no_ans_boost: float
        :param context_window_size: The size, in characters, of the window around the answer span that is used when displaying the context around the answer.
        :type context_window_size: int
        :param n_best: The number of positive answer spans for each document.
        :type n_best: int
        :param n_best_per_sample: num candidate answer spans to consider from each passage. Each passage also returns "no answer" info.
                                  This is decoupled from n_best on document level, since predictions on passage level are very similar.
                                  It should have a low value
        :type n_best_per_sample: int
        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.
                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
        :type duplicate_filtering: int
        """
        super(QuestionAnsweringHead, self).__init__()
        if len(kwargs) > 0:
            logger.warning(f"Some unused parameters are passed to the QuestionAnsweringHead. "
                           f"Might not be a problem. Params: {json.dumps(kwargs)}")
        self.layer_dims = layer_dims
        assert self.layer_dims[-1] == 2
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_token_squad"
        self.model_type = ("span_classification")  # predicts start and end token of answer
        self.task_name = task_name
        self.no_ans_boost = no_ans_boost
        self.context_window_size = context_window_size
        self.n_best = n_best
        self.n_best_per_sample = n_best_per_sample
        self.duplicate_filtering = duplicate_filtering
        self.generate_config()


    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - bert-large-uncased-whole-word-masking-finetuned-squad

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            super(QuestionAnsweringHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_qa_model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, revision=revision)
            # init empty head
            head = cls(layer_dims=[full_qa_model.config.hidden_size, 2], task_name="question_answering")
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_qa_model.qa_outputs.state_dict())
            del full_qa_model

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

    def logits_to_preds(self, logits, span_mask, start_of_word,
                        seq_2_start_t, max_answer_length=1000, **kwargs):
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

        # get scores for all combinations of start and end logits => candidate answers
        start_matrix = start_logits.unsqueeze(2).expand(-1, -1, max_seq_len)
        end_matrix = end_logits.unsqueeze(1).expand(-1, max_seq_len, -1)
        start_end_matrix = start_matrix + end_matrix

        # disqualify answers where end < start
        # (set the lower triangular matrix to low value, excluding diagonal)
        indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=start_end_matrix.device)
        start_end_matrix[:, indices[0][:], indices[1][:]] = -888

        # disqualify answers where answer span is greater than max_answer_length
        # (set the upper triangular matrix to low value, excluding diagonal)
        indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=max_answer_length, device=start_end_matrix.device)
        start_end_matrix[:, indices_long_span[0][:], indices_long_span[1][:]] = -777

        # disqualify answers where start=0, but end != 0
        start_end_matrix[:, 0, 1:] = -666

        # Turn 1d span_mask vectors into 2d span_mask along 2 different axes
        # span mask has:
        #   0 for every position that is never a valid start or end index (question tokens, mid and end special tokens, padding)
        #   1 everywhere else
        span_mask_start = span_mask.unsqueeze(2).expand(-1, -1, max_seq_len)
        span_mask_end = span_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        span_mask_2d = span_mask_start + span_mask_end
        # disqualify spans where either start or end is on an invalid token
        invalid_indices = torch.nonzero((span_mask_2d != 2), as_tuple=True)
        start_end_matrix[invalid_indices[0][:], invalid_indices[1][:], invalid_indices[2][:]] = -999

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

        # Get the n_best candidate answers for each sample
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(sorted_candidates[sample_idx],
                                                   start_end_matrix[sample_idx],
                                                   sample_idx)
            all_top_n.append(sample_top_n)

        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix, sample_idx):
        """ Returns top candidate answers as a list of Span objects. Operates on a matrix of summed start and end logits.
        This matrix corresponds to a single sample (includes special tokens, question tokens, passage tokens).
        This method always returns a list of len n_best + 1 (it is comprised of the n_best positive answers along with the one no_answer)"""

        # Initialize some variables
        top_candidates = []
        n_candidates = sorted_candidates.shape[0]
        start_idx_candidates = set()
        end_idx_candidates = set()

        # Iterate over all candidates and break when we have all our n_best candidates
        for candidate_idx in range(n_candidates):
            if len(top_candidates) == self.n_best_per_sample:
                break
            else:
                # Retrieve candidate's indices
                start_idx = sorted_candidates[candidate_idx, 0].item()
                end_idx = sorted_candidates[candidate_idx, 1].item()
                # Ignore no_answer scores which will be extracted later in this method
                if start_idx == 0 and end_idx == 0:
                    continue
                if self.duplicate_filtering > -1 and (start_idx in start_idx_candidates or end_idx in end_idx_candidates):
                    continue
                score = start_end_matrix[start_idx, end_idx].item()
                top_candidates.append(QACandidate(offset_answer_start=start_idx,
                                                  offset_answer_end=end_idx,
                                                  score=score,
                                                  answer_type="span",
                                                  offset_unit="token",
                                                  aggregation_level="passage",
                                                  passage_id=sample_idx))
                if self.duplicate_filtering > -1:
                    for i in range(0, self.duplicate_filtering + 1):
                        start_idx_candidates.add(start_idx + i)
                        start_idx_candidates.add(start_idx - i)
                        end_idx_candidates.add(end_idx + i)
                        end_idx_candidates.add(end_idx - i)


        no_answer_score = start_end_matrix[0, 0].item()
        top_candidates.append(QACandidate(offset_answer_start=0,
                                          offset_answer_end=0,
                                          score=no_answer_score,
                                          answer_type="no_answer",
                                          offset_unit="token",
                                          aggregation_level="passage",
                                          passage_id=None))

        return top_candidates

    def formatted_preds(self, logits=None, preds=None, baskets=None, **kwargs):
        """ Takes a list of passage level predictions, each corresponding to one sample, and converts them into document level
        predictions. Leverages information in the SampleBaskets. Assumes that we are being passed predictions from
        ALL samples in the one SampleBasket i.e. all passages of a document. Logits should be None, because we have
        already converted the logits to predictions before calling formatted_preds.
        (see Inferencer._get_predictions_and_aggregate()).
        """

        # Unpack some useful variables
        # passage_start_t is the token index of the passage relative to the document (usually a multiple of doc_stride)
        # seq_2_start_t is the token index of the first token in passage relative to the input sequence (i.e. number of
        # special tokens and question tokens that come before the passage tokens)
        if logits or preds is None:
            logger.error("QuestionAnsweringHead.formatted_preds() expects preds as input and logits to be None \
                            but was passed something different")
        samples = [s for b in baskets for s in b.samples]
        ids = [s.id for s in samples]
        passage_start_t = [s.features[0]["passage_start_t"] for s in samples]
        seq_2_start_t = [s.features[0]["seq_2_start_t"] for s in samples]

        # Aggregate passage level predictions to create document level predictions.
        # This method assumes that all passages of each document are contained in preds
        # i.e. that there are no incomplete documents. The output of this step
        # are prediction spans
        preds_d = self.aggregate_preds(preds, passage_start_t, ids, seq_2_start_t)

        # Separate top_preds list from the no_ans_gap float.
        top_preds, no_ans_gaps = zip(*preds_d)

        # Takes document level prediction spans and returns string predictions
        doc_preds = self.to_qa_preds(top_preds, no_ans_gaps, baskets)

        return doc_preds

    def to_qa_preds(self, top_preds, no_ans_gaps, baskets):
        """ Groups Span objects together in a QAPred object  """
        ret = []

        # Iterate over each set of document level prediction
        for pred_d, no_ans_gap, basket in zip(top_preds, no_ans_gaps, baskets):

            # Unpack document offsets, clear text and id
            token_offsets = basket.raw["document_offsets"]
            pred_id = basket.id_external if basket.id_external else basket.id_internal

            # These options reflect the different input dicts that can be assigned to the basket
            # before any kind of normalization or preprocessing can happen
            question_names = ["question_text", "qas", "questions"]
            doc_names = ["document_text", "context", "text"]

            document_text = try_get(doc_names, basket.raw)
            question = self.get_question(question_names, basket.raw)
            ground_truth = self.get_ground_truth(basket)

            curr_doc_pred = QAPred(id=pred_id,
                                   prediction=pred_d,
                                   context=document_text,
                                   question=question,
                                   token_offsets=token_offsets,
                                   context_window_size=self.context_window_size,
                                   aggregation_level="document",
                                   ground_truth_answer=ground_truth,
                                   no_answer_gap=no_ans_gap)

            ret.append(curr_doc_pred)
        return ret

    @staticmethod
    def get_ground_truth(basket):
        if "answers" in basket.raw:
            return basket.raw["answers"]
        elif "annotations" in basket.raw:
            return basket.raw["annotations"]
        else:
            return None

    @staticmethod
    def get_question(question_names, raw_dict):
        # For NQ style dicts
        qa_name = None
        if "qas" in raw_dict:
            qa_name = "qas"
        elif "question" in raw_dict:
            qa_name = "question"
        if qa_name:
            if type(raw_dict[qa_name][0]) == dict:
                return raw_dict[qa_name][0]["question"]
        return try_get(question_names, raw_dict)

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

        # Iterate over the preds of each sample - remove final number which is the sample id and not needed for aggregation
        for sample_idx in range(n_samples):
            basket_id = ids[sample_idx]
            basket_id = basket_id.split("-")[:-1]
            basket_id = "-".join(basket_id)

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

    def reduce_preds(self, preds):
        """ This function contains the logic for choosing the best answers from each passage. In the end, it
        returns the n_best predictions on the document level. """

        # Initialize variables
        passage_no_answer = []
        passage_best_score = []
        no_answer_scores = []
        n_samples = len(preds)

        # Iterate over the top predictions for each sample
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred.score
            no_answer_score = self.get_no_answer_score(sample_preds) + self.no_ans_boost
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            passage_best_score.append(best_pred_score)

        # Get all predictions in flattened list and sort by score
        pos_answers_flat = []
        for sample_idx, passage_preds in enumerate(preds):
            for qa_candidate in passage_preds:
                if not (qa_candidate.offset_answer_start == -1 and qa_candidate.offset_answer_end == -1):
                    pos_answers_flat.append(QACandidate(offset_answer_start=qa_candidate.offset_answer_start,
                                                        offset_answer_end=qa_candidate.offset_answer_end,
                                                        score=qa_candidate.score,
                                                        answer_type=qa_candidate.answer_type,
                                                        offset_unit="token",
                                                        aggregation_level="passage",
                                                        passage_id=str(sample_idx),
                                                        n_passages_in_doc=n_samples)
                                            )

        # TODO add switch for more variation in answers, e.g. if varied_ans then never return overlapping answers
        pos_answer_dedup = self.deduplicate(pos_answers_flat)

        # This is how much no_ans_boost needs to change to turn a no_answer to a positive answer (or vice versa)
        no_ans_gap = -min([nas - pbs for nas, pbs in zip(no_answer_scores, passage_best_score)])

        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to a specific text qa_candidate
        # - a "no answer" score is related to all input texts
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: change top prediction from "no answer" to answer (or vice versa)
        best_overall_positive_score = max(x.score for x in pos_answer_dedup)
        no_answer_pred = QACandidate(offset_answer_start=-1,
                                     offset_answer_end=-1,
                                     score=best_overall_positive_score - no_ans_gap,
                                     answer_type="no_answer",
                                     offset_unit="token",
                                     aggregation_level="document",
                                     passage_id=None,
                                     n_passages_in_doc=n_samples)

        # Add no answer to positive answers, sort the order and return the n_best
        n_preds = [no_answer_pred] + pos_answer_dedup
        n_preds_sorted = sorted(n_preds, key=lambda x: x.score, reverse=True)
        n_preds_reduced = n_preds_sorted[:self.n_best]
        return n_preds_reduced, no_ans_gap

    @staticmethod
    def deduplicate(flat_pos_answers):
        # Remove duplicate spans that might be twice predicted in two different passages
        seen = {}
        for qa_answer in flat_pos_answers:
            if (qa_answer.offset_answer_start, qa_answer.offset_answer_end) not in seen:
                seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)] = qa_answer
            else:
                seen_score = seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)].score
                if qa_answer.score > seen_score:
                    seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)] = qa_answer
        return list(seen.values())


    @staticmethod
    def get_no_answer_score(preds):
        for qa_answer in preds:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            score = qa_answer.score
            if start == -1 and end == -1:
                return score
        raise Exception

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t):
        """ Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) qa_answer but will instead be represented by (-1, -1)"""
        new_pred = []
        for qa_answer in pred:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                if start < 0:
                    logger.error("Start token index < 0 (document level)")
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                if end < 0:
                    logger.error("End token index < 0 (document level)")
            qa_answer.to_doc_level(start, end)
            new_pred.append(qa_answer)
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

    @staticmethod
    def merge_formatted_preds(preds_all):
        """ Merges results from the two prediction heads used for NQ style QA. Takes the prediction from QA head and
        assigns it the appropriate classification label. This mapping is achieved through passage_id.
        preds_all should contain [QuestionAnsweringHead.formatted_preds(), TextClassificationHead()]. The first item
        of this list should be of len=n_documents while the second item should be of len=n_passages"""

        ret = []

        # This fn is used to align QA output of len=n_docs and Classification output of len=n_passages
        def chunk(iterable, lengths):
            if sum(lengths) != len(iterable):
                logger.error("Sum of the lengths does not match the length of the iterable")
            cumsum = list(np.cumsum(lengths))
            cumsum = [0] + cumsum
            spans = [(cumsum[i], cumsum[i+1]) for i in range(len(cumsum) - 1)]
            ret = [iterable[start:end] for start, end in spans]
            return ret

        cls_preds = preds_all[1][0]["predictions"]
        qa_preds = preds_all[0][0]
        samples_per_doc = [doc_pred.n_passages for doc_pred in preds_all[0][0]]
        cls_preds_grouped = chunk(cls_preds, samples_per_doc)

        for qa_pred, cls_preds in zip(qa_preds, cls_preds_grouped):
            qa_candidates = qa_pred.prediction
            qa_candidates_new = []
            for qa_candidate in qa_candidates:
                passage_id = qa_candidate.passage_id
                if passage_id is not None:
                    cls_pred = cls_preds[int(passage_id)]["label"]
                # i.e. if no_answer
                else:
                    cls_pred = "no_answer"
                qa_candidate.add_cls(cls_pred)
                qa_candidates_new.append(qa_candidate)
            qa_pred.prediction = qa_candidates_new
            ret.append(qa_pred)
        return ret


def pick_single_fn(heads, fn_name):
    """ Iterates over heads and returns a static method called fn_name
    if and only if one head has a method of that name. If no heads have such a method, None is returned.
    If more than one head has such a method, an Exception is thrown"""
    merge_fns = []
    for h in heads:
        merge_fns.append(getattr(h, fn_name, None))

    merge_fns = [x for x in merge_fns if x is not None]
    if len(merge_fns) == 0:
        return None
    elif len(merge_fns) == 1:
        return merge_fns[0]
    else:
        raise Exception(f"More than one of the prediction heads have a {fn_name}() function")


class TextSimilarityHead(PredictionHead):
    """
    Trains a head on predicting the similarity of two texts like in Dense Passage Retrieval.
    """
    def __init__(self, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000, **kwargs):
        """
        Init the TextSimilarityHead.

        :param similarity_function: Function to calculate similarity between queries and passage embeddings.
                                    Choose either "dot_product" (Default) or "cosine".
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up

        :param kwargs:
        """

        super(TextSimilarityHead, self).__init__()

        self.similarity_function = similarity_function
        self.loss_fct = NLLLoss(reduction="mean")
        self.task_name = "text_similarity"
        self.model_type = "text_similarity"
        self.ph_output_type = "per_sequence"
        self.global_loss_buffer_size = global_loss_buffer_size
        self.generate_config()

    @classmethod
    def dot_product_scores(cls, query_vectors, passage_vectors):
        """
        Calculates dot product similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model of dimension n1 x D, where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model of dimension n2 x D, where n2 is the number of queries/batch size and D is embedding size
        :type passage_vectors: torch.Tensor

        :return dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
        return dot_product

    @classmethod
    def cosine_scores(cls, query_vectors, passage_vectors):
        """
        Calculates cosine similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is the number of queries/batch size and D is embedding size
        :type passage_vectors: torch.Tensor

        :return: cosine similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        return nn.functional.cosine_similarity(query_vectors, passage_vectors, dim=1)

    def get_similarity_function(self):
        """
        Returns the type of similarity function used to compare queries and passages/contexts
        """
        if "dot_product" in self.similarity_function:
            return TextSimilarityHead.dot_product_scores
        elif "cosine" in self.similarity_function:
            return TextSimilarityHead.cosine_scores

    def forward(self, query_vectors:torch.Tensor, passage_vectors:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Only packs the embeddings from both language models into a tuple. No further modification.
        The similarity calculation is handled later to enable distributed training (DDP)
        while keeping the support for in-batch negatives.
        (Gather all embeddings from nodes => then do similarity scores + loss)

        :param query_vectors: Tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: Tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is the number of queries/batch size and D is embedding size
        :type passage_vectors: torch.Tensor

        :return: (query_vectors, passage_vectors)
        """
        return (query_vectors, passage_vectors)

    def _embeddings_to_scores(self, query_vectors:torch.Tensor, passage_vectors:torch.Tensor):
        """
        Calculates similarity scores between all given query_vectors and passage_vectors

        :param query_vectors: Tensor of queries encoded by the query encoder model
        :param passage_vectors: Tensor of passages encoded by the passage encoder model
        :return: Tensor of log softmax similarity scores of each query with each passage (dimension: n1xn2)
        """

        sim_func = self.get_similarity_function()
        scores = sim_func(query_vectors, passage_vectors)

        if len(query_vectors.size()) > 1:
            q_num = query_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = nn.functional.log_softmax(scores, dim=1)
        return softmax_scores

    def logits_to_loss(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        """
        Computes the loss (Default: NLLLoss) by applying a similarity function (Default: dot product) to the input
        tuple of (query_vectors, passage_vectors) and afterwards applying the loss function on similarity scores.

        :param logits: Tuple of Tensors (query_embedding, passage_embedding) as returned from forward()

        :return: negative log likelihood loss from similarity scores
        """

        # Check if DDP is initialized
        try:
            rank = torch.distributed.get_rank()
        except AssertionError:
            rank = -1

        # Prepare predicted scores
        query_vectors, passage_vectors = logits

        # Prepare Labels
        lm_label_ids = kwargs.get(self.label_tensor_name)
        positive_idx_per_question = torch.nonzero((lm_label_ids.view(-1) == 1), as_tuple=False)

        # Gather global embeddings from all distributed nodes (DDP)
        if rank != -1:
            q_vector_to_send = torch.empty_like(query_vectors).cpu().copy_(query_vectors).detach_()
            p_vector_to_send = torch.empty_like(passage_vectors).cpu().copy_(passage_vectors).detach_()

            global_question_passage_vectors = all_gather_list(
                [q_vector_to_send, p_vector_to_send, positive_idx_per_question],
                max_size=self.global_loss_buffer_size)

            global_query_vectors = []
            global_passage_vectors = []
            global_positive_idx_per_question = []
            total_passages = 0
            for i, item in enumerate(global_question_passage_vectors):
                q_vector, p_vectors, positive_idx = item

                if i != rank:
                    global_query_vectors.append(q_vector.to(query_vectors.device))
                    global_passage_vectors.append(p_vectors.to(passage_vectors.device))
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx])
                else:
                    global_query_vectors.append(query_vectors)
                    global_passage_vectors.append(passage_vectors)
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx_per_question])
                total_passages += p_vectors.size(0)

            global_query_vectors = torch.cat(global_query_vectors, dim=0)
            global_passage_vectors = torch.cat(global_passage_vectors, dim=0)
            global_positive_idx_per_question = torch.LongTensor(global_positive_idx_per_question)
        else:
            global_query_vectors = query_vectors
            global_passage_vectors = passage_vectors
            global_positive_idx_per_question = positive_idx_per_question

        # Get similarity scores
        softmax_scores = self._embeddings_to_scores(global_query_vectors, global_passage_vectors)
        targets = global_positive_idx_per_question.squeeze(-1).to(softmax_scores.device)

        # Calculate loss
        loss = self.loss_fct(softmax_scores, targets)
        return loss

    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        """
        Returns predicted ranks(similarity) of passages/context for each query

        :param logits: tensor of log softmax similarity scores of each query with each context/passage (dimension: n1xn2)
        :type logits: torch.Tensor

        :return: predicted ranks of passages for each query
        """
        query_vectors, passage_vectors = logits
        softmax_scores = self._embeddings_to_scores(query_vectors, passage_vectors)
        _, sorted_scores = torch.sort(softmax_scores, dim=1, descending=True)
        return sorted_scores

    def prepare_labels(self, **kwargs):
        """
        Returns a tensor with passage labels(0:hard_negative/1:positive) for each query

        :return: passage labels(0:hard_negative/1:positive) for each query
        """
        label_ids = kwargs.get(self.label_tensor_name)
        labels = torch.zeros(label_ids.size(0), label_ids.numel())
        
        positive_indices = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)

        for i, indx in enumerate(positive_indices):
            labels[i, indx.item()] = 1
        return labels

    def formatted_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        raise NotImplementedError("formatted_preds is not supported in TextSimilarityHead yet!")
