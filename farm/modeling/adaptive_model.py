import logging
import os

from torch import nn

from farm.file_utils import create_folder
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import PredictionHead
from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


class AdaptiveModel(nn.Module):
    """ Combines a language model and a prediction head for an NLP task. Allows for gradient
    flow back to the language model component"""

    def __init__(
        self,
        language_model,
        prediction_heads,
        embeds_dropout_prob,
        lm_output_types,
        device,
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
        to "per_token", one embedding will be extracted per input token. If set to "per_sequence", a single embedding
        will be extracted to represent the full input sequence. Can either be a single string, or a list of strings,
        one for each prediction head.
        :type lm_output_types: list or str
        :param device: The device on which this model will operate. Either "cpu" or "cuda".
        """
        super(AdaptiveModel, self).__init__()
        self.language_model = language_model.to(device)
        self.prediction_heads = [ph.to(device) for ph in prediction_heads]
        self.num_labels = [head.num_labels for head in prediction_heads]
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = (
            [lm_output_types] if isinstance(lm_output_types, str) else lm_output_types
        )

        self.log_params()

    def save(self, save_dir):
        create_folder(save_dir)
        self.language_model.save(save_dir)
        for i, ph in enumerate(self.prediction_heads):
            ph.save(save_dir, i)
            # Need to save config and pipeline

    @classmethod
    def load(cls, load_dir, device):
        # Prediction heads
        ph_model_files, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for model_file, config_file in zip(ph_model_files, ph_config_files):
            head = PredictionHead.load(
                model_file=model_file, config_file=config_file, device=device
            )
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        # Language Model
        language_model = LanguageModel.load(load_dir)

        return cls(language_model, prediction_heads, 0.1, ph_output_type, device)

    def logits_to_loss_per_head(self, logits, **kwargs):
        # collect losses from all heads
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits, **kwargs):
        # get losses from all heads & reduce to single loss *per sample*
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        loss = sum(all_losses)
        return loss

    def logits_to_preds(self, logits, label_maps, **kwargs):
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head, label_map_for_head in zip(
            self.prediction_heads, logits, label_maps
        ):
            preds = head.logits_to_preds(
                logits=logits_for_head, label_map=label_map_for_head, **kwargs
            )
            all_preds.append(preds)
        return all_preds

    def prepare_labels(self, label_maps, **kwargs):
        all_labels = []
        for head, label_map_one_head in zip(self.prediction_heads, label_maps):
            labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
            all_labels.append(labels)
        return all_labels

    def formatted_preds(self, logits, label_maps, **kwargs):
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head, label_map_for_head in zip(
            self.prediction_heads, logits, label_maps
        ):
            preds = head.formatted_preds(
                logits=logits_for_head, label_map=label_map_for_head, **kwargs
            )
            all_preds.append(preds)
        return all_preds

    def forward(self, **kwargs):
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
            elif lm_out == "per_sequence":
                output = self.dropout(pooled_output)
            elif lm_out == "per_token_squad":
                output = self.dropout(sequence_output)
            else:
                raise ValueError(
                    "Unknown extraction strategy from language model: {}".format(lm_out)
                )

            # Do the actual forward pass of a single head
            all_logits.append(head(output))

        return all_logits

    @classmethod
    def _get_prediction_head_files(cls, load_dir):
        files = os.listdir(load_dir)
        model_files = [
            os.path.join(load_dir, f)
            for f in files
            if ".bin" in f and "prediction_head" in f
        ]
        config_files = [
            os.path.join(load_dir, f)
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
        params = {
            "lm": self.language_model.__class__.__name__,
            "prediction_heads": ",".join(
                [head.__class__.__name__ for head in self.prediction_heads]
            ),
            "lm_output_types": ",".join(self.lm_output_types),
        }
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")
