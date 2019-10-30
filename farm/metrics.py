import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import f1_score as ner_f1_score
from sklearn.metrics import matthews_corrcoef, f1_score, mean_squared_error, r2_score
from farm.utils import flatten_list
import logging

logger = logging.getLogger(__name__)

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
    if not metric in ["squad","squad_top_recall"]:
        assert len(preds) == len(labels)
    if metric == "mcc":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif metric == "acc":
        return simple_accuracy(preds, labels)
    elif metric == "acc_f1":
        return acc_and_f1(preds, labels)
    elif metric == "pear_spear":
        return pearson_and_spearman(preds, labels)
    # TODO this metric seems very specific for NER and doesnt work for other sequence labeling tasks
    elif metric == "seq_f1":
        return {"seq_f1": ner_f1_score(labels, preds)}
    elif metric == "f1_macro":
        return f1_macro(preds, labels)
    elif metric == "squad":
        return squad(preds, labels)
    elif metric == "squad_top_recall":
        return squad_N_recall(preds=preds, labels=labels)
    elif metric == "mse":
        return {"mse": mean_squared_error(preds, labels)}
    elif metric == "r2":
        return {"r2": r2_score(preds, labels)}
    # elif metric == "masked_accuracy":
    #     return simple_accuracy(preds, labels, ignore=-1)
    else:
        raise KeyError(metric)

def squad_N_recall(preds=None, labels=None):
    # checks weather any one token is within ground truth answer
    # can be used to only check weather QA model finds the right location
    # should be used when answering on long documents and difficult questions
    success_all = []
    data = {}
    data["sample_id"] = np.concatenate([x[0].cpu().numpy() for x in labels])
    data["start_idx"] = np.concatenate([x[1].cpu().numpy() for x in labels])
    data["end_idx"] = np.concatenate([x[2].cpu().numpy() for x in labels])
    df = pd.DataFrame(data=data)

    for max_pred in preds:
        sampled_id = max_pred[0,3]
        label_group = df.loc[df.sample_id == sampled_id, :]
        label = label_group.loc[label_group.end_idx == np.max(label_group.end_idx), ["start_idx","end_idx"]].values
        success = 0
        for i in range(max_pred.shape[0]):
            em,p,r,f1 = compute_qa_f1(pred_start=max_pred[i,0],
                                      pred_end=max_pred[i,1],
                                      label_start=label[0,0],
                                      label_end=label[0,1])
            if(r > 0):
                success = 1
        success_all.append(success)
    return {f"TopN Passage Recall": np.mean(success_all)}

def squad(preds=None, labels=None):
    # scoring in tokenized space and with only one answer per sample (in squad dev set multiple answers are given).
    # So results to official evaluation will vary
    em_all = []
    f1_all = []
    precision_all = []
    recall_all = []
    data = {}
    data["sample_id"] = np.concatenate([x[0].cpu().numpy() for x in labels])
    data["start_idx"] = np.concatenate([x[1].cpu().numpy() for x in labels])
    data["end_idx"] = np.concatenate([x[2].cpu().numpy() for x in labels])
    df = pd.DataFrame(data=data)

    for max_pred in preds:
        sampled_id = max_pred[0,3]
        label_group = df.loc[df.sample_id == sampled_id, :]
        label = label_group.loc[label_group.end_idx == np.max(label_group.end_idx), ["start_idx","end_idx"]].values
        em,p,r,f1 = compute_qa_f1(pred_start=max_pred[0,0],
                                  pred_end=max_pred[0,1],
                                  label_start=label[0,0],
                                  label_end=label[0,1])
        em_all.append(em)
        precision_all.append(p)
        recall_all.append(r)
        f1_all.append(f1)


    return {"EM": np.mean(em_all),
            "Precision": np.mean(precision_all),
            "Recall": np.mean(recall_all),
            "F1": np.mean(f1_all)}


def compute_qa_f1(pred_start, pred_end, label_start, label_end):
    # scoring in tokenized space, so results to public leaderboard will vary
    em = pred_start == label_start and pred_end == label_end
    if (pred_start + pred_end) <= 0 or (label_start + label_end) <= 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        f1 = pred_end == label_end
        precision = pred_end == label_end
        recall = pred_end == label_end
    else:
        pred_range = set(range(pred_start, pred_end + 1)) # include end pred
        true_range = set(range(label_start, label_end + 1))
        num_same = len(true_range.intersection(pred_range))
        if num_same == 0:
            f1 = 0
            precision = 0
            recall = 0
        else:
            precision = 1.0 * num_same / len(pred_range)
            recall = 1.0 * num_same / len(true_range)
            f1 = (2 * precision * recall) / (precision + recall)
    return em,precision, recall, f1

