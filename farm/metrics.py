import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import f1_score as ner_f1_score
from sklearn.metrics import matthews_corrcoef, f1_score, mean_squared_error, r2_score
from farm.utils import flatten_list

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
    if not metric == "squad":
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
    elif metric == "mse":
        return {"mse": mean_squared_error(preds, labels)}
    elif metric == "r2":
        return {"r2": r2_score(preds, labels)}
    # elif metric == "masked_accuracy":
    #     return simple_accuracy(preds, labels, ignore=-1)
    else:
        raise KeyError(metric)


def squad_EM(preds, labels):
    # scoring in tokenized space, so results to public leaderboard will vary
    pred_start = np.concatenate(preds[::3])
    pred_end = np.concatenate(preds[1::3])
    label_start = torch.cat(labels[::2])
    label_end = torch.cat(labels[1::2])
    assert len(label_start) == len(pred_start)
    num_total = len(label_start)
    num_correct = 0
    for i in range(num_total):
        if pred_start[i] == label_start[i] and pred_end[i] == label_end[i]:
            num_correct += 1
    return num_correct / num_total


def squad_f1(preds, labels):
    # scoring in tokenized space, so results to public leaderboard will vary
    pred_start = np.concatenate(preds[::3]) # having start, end and probabilities in preds
    pred_end = np.concatenate(preds[1::3])

    label_start = torch.cat(labels[::2]).cpu().numpy()
    label_end = torch.cat(labels[1::2]).cpu().numpy()
    assert len(label_start) == len(pred_start)
    num_total = len(label_start)
    f1_scores = []
    prec_scores = []
    recall_scores = []
    for i in range(num_total):
        if (pred_start[i] + pred_end[i]) <= 0 or (label_start[i] + label_end[i]) <= 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            f1_scores.append(pred_end[i] == label_end[i])
            prec_scores.append(pred_end[i] == label_end[i])
            recall_scores.append(pred_end[i] == label_end[i])
        else:
            pred_range = set(range(pred_start[i], pred_end[i]))
            true_range = set(range(label_start[i], label_end[i]))
            num_same = len(true_range.intersection(pred_range))
            if num_same == 0:
                f1_scores.append(0)
                prec_scores.append(0)
                recall_scores.append(0)
            else:
                precision = 1.0 * num_same / len(pred_range)
                recall = 1.0 * num_same / len(true_range)
                f1 = (2 * precision * recall) / (precision + recall)
                f1_scores.append(f1)
                prec_scores.append(precision)
                recall_scores.append(recall)
    return (
        np.mean(np.array(prec_scores)),
        np.mean(np.array(recall_scores)),
        np.mean(np.array(f1_scores)),
    )


def squad(preds, labels):
    em = squad_EM(preds=preds, labels=labels)
    f1 = squad_f1(preds=preds, labels=labels)

    return {"EM": em, "f1": f1}
