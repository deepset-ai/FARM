import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class AdaptiveModel(nn.Module):
    """ Combines a language model and a prediction head for an NLP task. Allows for gradient
    flow back to the language model component"""
    def __init__(self, language_model, prediction_head, embeds_dropout_prob, token_level):
        super(AdaptiveModel, self).__init__()
        self.language_model = language_model
        self.prediction_head = prediction_head
        self.num_labels = prediction_head.num_labels
        self.dropout = nn.Dropout(embeds_dropout_prob)
        # TODO: RENAME
        self.token_level = token_level

    def save(self, save_dir):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_dir):
        raise NotImplementedError()

    def logits_to_loss(self, logits, labels, **kwargs):
        return self.prediction_head.logits_to_loss(logits, labels, **kwargs)

    def logits_to_preds(self, logits, **kwargs):
        return self.prediction_head.logits_to_preds(logits, **kwargs)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None):

        sequence_output, pooled_output = self.language_model(input_ids,
                                               token_type_ids,
                                               attention_mask,
                                               output_all_encoded_layers=False)

        if self.token_level:
            output = sequence_output
        else:
            output = pooled_output

        output = self.dropout(output)
        logits = self.prediction_head(output)

        return logits


class NERClassifier(AdaptiveModel):
    """ A Classifier that performs Named Entity Recognition. Can handle mapping of word to tokens """
    def __init__(self,
                 language_model,
                 prediction_head,
                 embeds_dropout_prob,
                 balanced_weights=None):

        super(NERClassifier, self).__init__(language_model, prediction_head)


        self.language_model = language_model
        self.prediction_head = prediction_head

        self.num_labels = prediction_head.output_size

        self.dropout = nn.Dropout(embeds_dropout_prob)

        # needs to be a parameter for distributed setting
        # This is messy, can we do this differently?
        if balanced_weights:
            self.balanced_weights = nn.Parameter(torch.tensor(balanced_weights), requires_grad=False)
            self.loss_fct = CrossEntropyLoss(weight=self.balanced_weights, reduction="none")
        else:
            self.loss_fct = CrossEntropyLoss(reduction="none")

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None):

        sequence_output, _ = self.language_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.prediction_head(sequence_output)
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
        input_mask = input_mask.detach().cpu().numpy()
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
