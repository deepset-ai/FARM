from torch import nn
from farm.utils import MLFlowLogger as MlLogger
from farm.modeling.prediction_head import PredictionHead


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
        super(AdaptiveModel, self).__init__()
        self.language_model = language_model.to(device)
        self.prediction_heads = [ph.to(device) for ph in prediction_heads]
        self.num_labels = [head.num_labels for head in prediction_heads]
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = lm_output_types

        self.log_params()

    def save(self, save_dir):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_dir):
        raise NotImplementedError()

    def logits_to_loss_per_head(self, logits, **kwargs):
        # collect losses from all heads
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            all_losses.append(head.logits_to_loss(logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits, **kwargs):
        # get losses from all heads & reduce to single loss *per sample*
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        loss = sum(all_losses)
        return loss

    def logits_to_preds(self, logits, **kwargs):
        all_preds = []
        # collect preds from all heads
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits_for_one_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def prepare_labels(self, **kwargs):
        all_labels = []
        # if type(label_ids) != list:
        #   label_ids = [label_ids]
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

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
            else:
                raise ValueError(
                    "Unknown extraction strategy from language model: {}".format(lm_out)
                )

            # Do the actual forward pass of a single head
            all_logits.append(head(output))

        return all_logits

    def log_params(self):
        params = {
            "lm": self.language_model.__class__.__name__,
            "prediction_heads": ",".join(
                [head.__class__.__name__ for head in self.prediction_heads]
            ),
            "lm_output_types": ",".join(self.lm_output_types),
        }
        MlLogger.log_params(params)
