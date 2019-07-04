import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class AdaptiveModel(nn.Module):
    """ Combines a language model and a prediction head for an NLP task. Allows for gradient
    flow back to the language model component"""

    def __init__(
        self, language_model, prediction_head, embeds_dropout_prob, lm_output_type
    ):
        super(AdaptiveModel, self).__init__()
        self.language_model = language_model
        self.prediction_head = prediction_head
        self.num_labels = prediction_head.num_labels
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_type = lm_output_type

    def save(self, save_dir):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_dir):
        raise NotImplementedError()

    def logits_to_loss(self, logits, **kwargs):
        return self.prediction_head.logits_to_loss(logits, **kwargs)

    def logits_to_preds(self, logits, **kwargs):
        return self.prediction_head.logits_to_preds(logits, **kwargs)

    def forward(self, **kwargs):
        # Run language model
        sequence_output, pooled_output = self.language_model(
            **kwargs, output_all_encoded_layers=False
        )

        # Choose relevant vectors as output and perform dropout
        if self.lm_output_type == "per_token":
            output = self.dropout(sequence_output)
        elif self.lm_output_type == "per_sequence":
            output = self.dropout(pooled_output)
        elif self.lm_output_type == "both":
            output = [self.dropout(sequence_output), self.dropout(pooled_output)]
        else:
            raise ValueError("Unknown extraction strategy from language model")

        # Run prediction head
        logits = self.prediction_head(output)

        return logits
