import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import f1_score as ner_f1_score
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, f1_score, mean_squared_error, r2_score
from farm.utils import flatten_list
import logging

logger = logging.getLogger(__name__)

registered_metrics = {}

def register_metrics(name, implementation):
    registered_metrics[name] = implementation

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
    # elif metric == "masked_accuracy":
    #     return simple_accuracy(preds, labels, ignore=-1)
    elif metric in registered_metrics:
        metric_func = registered_metrics[metric]
        return metric_func(preds, labels)
    else:
        raise KeyError(metric)

def squad_EM(preds, labels):
    # TODO write comment describing function
    n_docs = len(preds)
    n_correct = 0
    for doc_idx in range(n_docs):
        pred_start, pred_end, _ = preds[doc_idx][0][0]
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


def squad_f1_single(pred, label):
    label_start, label_end = label
    pred_start, pred_end, _ = pred[0]
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

    return {"EM": em, "f1": f1}
