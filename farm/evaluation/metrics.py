import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report as token_classification_report
from seqeval.metrics import f1_score as ner_f1_score
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    mean_squared_error,
    r2_score,
    classification_report
)

from farm.utils import flatten_list

logger = logging.getLogger(__name__)

registered_metrics = {}
registered_reports = {}

def register_metrics(name, implementation):
    registered_metrics[name] = implementation

def register_report(name, implementation):
    """
    Register a custom reporting function to be used during eval.

    This can be useful:
    - if you want to overwrite a report for an existing output type of prediction head (e.g. "per_token")
    - if you have a new type of prediction head and want to add a custom report for it

    :param name: This must match the `ph_output_type` attribute of the PredictionHead for which the report should be used.
                 (e.g. TokenPredictionHead => `per_token`, YourCustomHead => `some_new_type`).
    :type name: str
    :param implementation: Function to be executed. It must take lists of `y_true` and `y_pred` as input and return a
                           printable object (e.g. string or dict).
                           See sklearns.metrics.classification_report for an example.
    :type implementation: function
    """
    registered_reports[name] = implementation

def simple_accuracy(preds, labels):
    # works also with nested lists of different lengths (needed for masked LM task)
    if type(preds) == type(labels) == list:
        preds = np.array(list(flatten_list(preds)))
        labels = np.array(list(flatten_list(labels)))
    assert type(preds) == type(labels) == np.ndarray
    correct = preds == labels
    return {"acc": correct.mean()}


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}


def f1_macro(preds, labels):
    return {"f1_macro": f1_score(y_true=labels, y_pred=preds, average="macro")}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(metric, preds, labels):
    assert len(preds) == len(labels)
    if metric == "mcc":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif metric == "acc":
        return simple_accuracy(preds, labels)
    elif metric == "acc_f1":
        return acc_and_f1(preds, labels)
    elif metric == "pear_spear":
        return pearson_and_spearman(preds, labels)
    # TODO this metric seems very specific for NER and doesnt work for
    elif metric == "seq_f1":
        return {"seq_f1": ner_f1_score(labels, preds)}
    elif metric == "f1_macro":
        return f1_macro(preds, labels)
    elif metric == "squad":
        return squad(preds, labels)
    elif metric == "mse":
        return {"mse": mean_squared_error(preds, labels)}
    elif metric == "r2":
        return {"r2": r2_score(preds, labels)}
    elif metric == "top_n_accuracy":
        return {"top_n_accuracy": top_n_accuracy(preds, labels)}
    # elif metric == "masked_accuracy":
    #     return simple_accuracy(preds, labels, ignore=-1)
    elif metric in registered_metrics:
        metric_func = registered_metrics[metric]
        return metric_func(preds, labels)
    else:
        raise KeyError(metric)


def compute_report_metrics(head, preds, labels):
    if head.ph_output_type in registered_reports:
        report_fn = registered_reports[head.ph_output_type]
    elif head.ph_output_type == "per_token":
        report_fn = token_classification_report
    elif head.ph_output_type == "per_sequence":
        report_fn = classification_report
    elif head.ph_output_type == "per_token_squad":
        report_fn = lambda *args, **kwargs: "Not Implemented"
    elif head.ph_output_type == "per_sequence_continuous":
        report_fn = r2_score
    else:
        raise AttributeError(f"No report function for head.ph_output_type '{head.ph_output_type}'. "
                             f"You can register a custom one via register_report(name='{head.ph_output_type}', implementation=<your_report_function>")

    # CHANGE PARAMETERS, not all report_fn accept digits
    if head.ph_output_type in ["per_sequence"]:
        # supply labels as all possible combination because if ground truth labels do not cover
        # all values in label_list (maybe dev set is small), the report will break
        if head.model_type == "multilabel_text_classification":
            # For multilabel classification, we don't eval with string labels here, but with multihot vectors.
            # Therefore we need to supply all possible label ids instead of label values.
            all_possible_labels = list(range(len(head.label_list)))
        else:
            all_possible_labels = head.label_list
        return report_fn(
            labels,
            preds,
            digits=4,
            labels=all_possible_labels,
            target_names=head.label_list
        )
    else:
        return report_fn(labels, preds)


def squad_EM(preds, labels):
    # TODO write comment describing function
    n_docs = len(preds)
    n_correct = 0
    for doc_idx in range(n_docs):
        qa_candidate = preds[doc_idx][0][0]
        pred_start = qa_candidate.offset_answer_start
        pred_end = qa_candidate.offset_answer_end
        curr_labels = labels[doc_idx]
        if (pred_start, pred_end) in curr_labels:
            n_correct += 1
    return n_correct/n_docs

def squad_f1(preds, labels):
    f1_scores = []
    n_docs = len(preds)
    for i in range(n_docs):
        best_pred = preds[i][0]
        best_f1 = max([squad_f1_single(best_pred, label) for label in labels[i]])
        f1_scores.append(best_f1)
    return np.mean(f1_scores)


def squad_f1_single(pred, label, pred_idx=0):
    label_start, label_end = label
    span = pred[pred_idx]
    pred_start = span.offset_answer_start
    pred_end = span.offset_answer_end

    if (pred_start + pred_end == 0) or (label_start + label_end == 0):
        if pred_start == label_start:
            return 1.0
        else:
            return 0.0
    pred_span = list(range(pred_start, pred_end + 1))
    label_span = list(range(label_start, label_end + 1))
    n_overlap = len([x for x in pred_span if x in label_span])
    if n_overlap == 0:
        return 0.0
    precision = n_overlap / len(pred_span)
    recall = n_overlap / len(label_span)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def squad(preds, labels):
    em = squad_EM(preds=preds, labels=labels)
    f1 = squad_f1(preds=preds, labels=labels)
    top_acc = top_n_accuracy(preds=preds, labels=labels)
    return {"EM": em, "f1": f1, "top_n_accuracy": top_acc}

def top_n_accuracy(preds, labels):
    answer_in_top_n = []
    n_questions = len(preds)
    for i in range(n_questions):
        f1_score = 0
        current_preds = preds[i][0]
        for idx, pred in enumerate(current_preds):
            f1_score = max([squad_f1_single(current_preds, label, pred_idx=idx) for label in labels[i]])
            if f1_score:
                break
        if f1_score:
            answer_in_top_n.append(1)
        else:
            answer_in_top_n.append(0)

    return np.mean(answer_in_top_n)



