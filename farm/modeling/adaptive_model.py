import logging
import os

from torch import nn
from pathlib import Path

from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import PredictionHead, BertLMHead, QuestionAnsweringHead, TokenClassificationHead, TextClassificationHead
from transformers.modeling_auto import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForTokenClassification
from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


class AdaptiveModel(nn.Module):
    """ Contains all the modelling needed for your NLP task. Combines a language model and a prediction head.
    Allows for gradient flow back to the language model component."""

    def __init__(
        self,
        language_model,
        prediction_heads,
        embeds_dropout_prob,
        lm_output_types,
        device,
        loss_aggregation_fn=None
    ):
        """
        :param language_model: Any model that turns token ids into vector representations
        :type language_model: LanguageModel
        :param prediction_heads: A list of models that take embeddings and return logits for a given task
        :type prediction_heads: list
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by the
           language model will be zeroed.
        :param embeds_dropout_prob: float
        :param lm_output_types: How to extract the embeddings from the final layer of the language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :type lm_output_types: list or str
        :param device: The device on which this model will operate. Either "cpu" or "cuda".
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
                                    Output: aggregated loss (tensor)
                                    Default is a simple sum:
                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
                                    However, you can pass more complex functions that depend on the
                                    current step (e.g. for round-robin style multitask learning) or the actual
                                    content of the batch (e.g. certain labels)
                                    Note: The loss at this stage is per sample, i.e one tensor of
                                    shape (batchsize) per prediction head.
        :type loss_aggregation_fn: function
        """

        super(AdaptiveModel, self).__init__()
        self.device = device
        self.language_model = language_model.to(device)
        self.lm_output_dims = language_model.get_output_dims()
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.fit_heads_to_lm()
        # set shared weights for LM finetuning
        for head in self.prediction_heads:
            if head.model_type == "language_modelling":
                head.set_shared_weights(language_model.model.embeddings.word_embeddings.weight)
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = (
            [lm_output_types] if isinstance(lm_output_types, str) else lm_output_types
        )
        self.log_params()
        # default loss aggregation function is a simple sum (without using any of the optional params)
        if not loss_aggregation_fn:
            loss_aggregation_fn = lambda loss_per_head, global_step=None, batch=None: sum(loss_per_head)
        self.loss_aggregation_fn = loss_aggregation_fn

    def fit_heads_to_lm(self):
        """This iterates over each prediction head and ensures that its input dimensionality matches the output
        dimensionality of the language model. If it doesn't, it is resized so it does fit"""
        for ph in self.prediction_heads:
            ph.resize_input(self.lm_output_dims)
            ph.to(self.device)

    def save(self, save_dir):
        """
        Saves the language model and prediction heads. This will generate a config file
        and model weights for each.

        :param save_dir: path to save to
        :type save_dir: Path
        """
        os.makedirs(save_dir, exist_ok=True)
        self.language_model.save(save_dir)
        for i, ph in enumerate(self.prediction_heads):
            ph.save(save_dir, i)
            # Need to save config and pipeline

    @classmethod
    def load(cls, load_dir, device, strict=True, lm_name=None):
        """
        Loads an AdaptiveModel from a directory. The directory must contain:

        * language_model.bin
        * language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Tokens

        :param load_dir: location where adaptive model is stored
        :type load_dir: Path
        :param device: to which device we want to sent the model, either cpu or cuda
        :type device: torch.device
        :param lm_name: the name to assign to the loaded language model
        :type lm_name: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        """

        # Language Model
        if lm_name:
            language_model = LanguageModel.load(load_dir, farm_lm_name=lm_name)
        else:
            language_model = LanguageModel.load(load_dir)

        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict)
            # # set shared weights between LM and PH
            # if type(head) == BertLMHead:
            #     head.set_shared_weights(language_model)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        return cls(language_model, prediction_heads, 0.1, ph_output_type, device)

    def logits_to_loss_per_head(self, logits, **kwargs):

        """
        Collect losses from each prediction head.

        :param logits: logits, can vary in shape and type, depending on task.
        :type logits: object
        :return: The per sample per prediciton head loss whose first two dimensions have length n_pred_heads, batch_size
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits, global_step=None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param global_step: number of current training step
        :type global_step: int
        :param kwargs: placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :type kwargs: object
        :return loss: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all predictions from all prediction heads
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        # for head, label_map_one_head in zip(self.prediction_heads):
        #     labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
        #     all_labels.append(labels)
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def formatted_preds(self, logits, **kwargs):
        """
        Format predictions for inference.

        :param logits: model logits
        :type logits: torch.tensor
        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(
            self.prediction_heads, logits
        ):
            preds = head.formatted_preds(
                logits=logits_for_head, **kwargs
            )
            all_preds.append(preds)
        return all_preds

    def forward(self, **kwargs):
        """
        Push data through the whole model and returns logits. The data will propagate through the language
        model and each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to the language model and prediction head(s).
        :return: all logits as torch.tensor or multiple tensors.
        """
        # Run language model
        sequence_output, pooled_output = self.language_model(
            **kwargs, output_all_encoded_layers=False
        )

        # Run (multiple) prediction heads
        all_logits = []
        for head, lm_out in zip(self.prediction_heads, self.lm_output_types):
            # Choose relevant vectors from LM as output and perform dropout
            if lm_out == "per_token":
                output = self.dropout(sequence_output)
            elif lm_out == "per_sequence" or lm_out == "per_sequence_continuous":
                output = self.dropout(pooled_output)
            elif (
                lm_out == "per_token_squad"
            ):  # we need a per_token_squad because of variable metric computation later on...
                output = self.dropout(sequence_output)
            else:
                raise ValueError(
                    "Unknown extraction strategy from language model: {}".format(lm_out)
                )

            # Do the actual forward pass of a single head
            all_logits.append(head(output))

        return all_logits

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """

        # Drop the next sentence prediction head if it does not appear in tasks. This is triggered by the interaction
        # setting the argument BertStyleLMProcessor(next_sent_pred=False)
        if "nextsentence" not in tasks:
            idx = None
            for i, ph in enumerate(self.prediction_heads):
                if ph.task_name == "nextsentence":
                    idx = i
            if idx is not None:
                logger.info(
                "Removing the NextSentenceHead since next_sent_pred is set to False in the BertStyleLMProcessor")
                del self.prediction_heads[i]

        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task \'{head.task_name}\' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            if "RegressionHead" in str(type(head)):
                # This needs to be explicitly set because the regression label_list is being hijacked to store
                # the scaling factor and the mean
                num_labels = 1
            else:
                num_labels = len(label_list)
            head.metric = tasks[head.task_name]["metric"]

    @classmethod
    def _get_prediction_head_files(cls, load_dir):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [
            load_dir / f
            for f in files
            if ".bin" in f and "prediction_head" in f
        ]
        config_files = [
            load_dir / f
            for f in files
            if "config.json" in f and "prediction_head" in f
        ]
        # sort them to get correct order in case of multiple prediction heads
        model_files.sort()
        config_files.sort()

        error_str = (
            "There is a mismatch in number of model files and config files. "
            "This might be because the Language Model Prediction Head "
            "does not currently support saving and loading"
        )
        assert len(model_files) == len(config_files), error_str
        logger.info(f"Found files for loading {len(model_files)} prediction heads")

        return model_files, config_files

    def log_params(self):
        """
        Logs paramteres to generic logger MlLogger
        """
        params = {
            "lm_type": self.language_model.__class__.__name__,
            "lm_name": self.language_model.name,
            "prediction_heads": ",".join(
                [head.__class__.__name__ for head in self.prediction_heads]
            ),
            "lm_output_types": ",".join(self.lm_output_types),
        }
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")

    def verify_vocab_size(self, vocab_size):
        """ Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()"""

        model_vocab_len = self.language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = f"Vocab size of tokenizer {vocab_size} doesn't match with model {model_vocab_len}. " \
              "If you added a custom vocabulary to the tokenizer, " \
              "make sure to supply 'n_added_tokens' to LanguageModel.load() and BertStyleLM.load()"
        assert vocab_size == model_vocab_len, msg

        for head in self.prediction_heads:
            if head.model_type == "language_modelling":
                ph_decoder_len = head.decoder.weight.shape[0]
                assert vocab_size == ph_decoder_len, msg

    def convert_to_transformers(self):
        if len(self.prediction_heads) != 1:
            raise ValueError(f"Currently conversion only works for models with a SINGLE prediction head. "
                             f"Your model has {len(self.prediction_heads)}")

        #TODO add more infos to config

        if self.prediction_heads[0].model_type == "span_classification":
            # init model
            transformers_model = AutoModelForQuestionAnswering.from_config(self.language_model.model.config)
            # transfer weights for language model + prediction head
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            transformers_model.qa_outputs.load_state_dict(
                self.prediction_heads[0].feed_forward.feed_forward[0].state_dict())

        elif self.prediction_heads[0].model_type == "text_classification":
            # add more info to config
            self.language_model.model.config.id2label = {id: label for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.label2id = {label: id for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.finetuning_task = "text_classification"
            self.language_model.model.config.language = self.language_model.language
            self.language_model.model.config.num_labels = self.prediction_heads[0].num_labels

            # init model
            transformers_model = AutoModelForSequenceClassification.from_config(self.language_model.model.config)
            # transfer weights for language model + prediction head
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            transformers_model.classifier.load_state_dict(
                self.prediction_heads[0].feed_forward.feed_forward[0].state_dict())
        elif self.prediction_heads[0].model_type == "token_classification":
            # add more info to config
            self.language_model.model.config.id2label = {id: label for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.label2id = {label: id for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.finetuning_task = "token_classification"
            self.language_model.model.config.language = self.language_model.language
            self.language_model.model.config.num_labels = self.prediction_heads[0].num_labels

            # init model
            transformers_model = AutoModelForTokenClassification.from_config(self.language_model.model.config)
            # transfer weights for language model + prediction head
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            transformers_model.classifier.load_state_dict(
                self.prediction_heads[0].feed_forward.feed_forward[0].state_dict())
        else:
            raise NotImplementedError(f"FARM -> Transformers conversion is not supported yet for"
                                      f" prediction heads of type {self.prediction_heads[0].model_type}")
        pass

        return transformers_model


    @classmethod
    def convert_from_transformers(cls, model_name_or_path, device, task_type):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in FARM (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path: local path of a saved model or name of a public one.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - deepset/bert-large-uncased-whole-word-masking-squad2

                                              See https://huggingface.co/models for full list
        :param device: "cpu" or "cuda"
        :param task_type: One of :
                          - 'question_answering'
                          - 'text_classification'
                          - 'embeddings'
                          More tasks coming soon ...
        :return: AdaptiveModel
        """
        lm = LanguageModel.load(model_name_or_path)
        #TODO Infer type of head automatically from config

        if task_type == "question_answering":
            ph = QuestionAnsweringHead.load(model_name_or_path)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                               lm_output_types="per_token", device=device)
        elif task_type == "text_classification":
            ph = TextClassificationHead.load(model_name_or_path)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                 lm_output_types="per_sequence", device=device)
        elif task_type == "ner":
            ph = TokenClassificationHead.load(model_name_or_path)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                               lm_output_types="per_token", device=device)
        elif task_type == "embeddings":
            adaptive_model = cls(language_model=lm, prediction_heads=[], embeds_dropout_prob=0.1,
                                 lm_output_types=["per_token", "per_sequence"], device=device)
        else:
            raise NotImplementedError(f"Huggingface's transformer models of type {task_type} are not supported yet")

        return adaptive_model
