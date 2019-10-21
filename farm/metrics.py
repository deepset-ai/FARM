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


def squad(preds=None, labels=None):
    # scoring in tokenized space, so results to public leaderboard will vary
    data = {}
    data["pred_start"] = np.concatenate(preds[::4])
    data["pred_end"] = np.concatenate(preds[1::4])
    data["prob"] = np.concatenate(preds[2::4])
    data["sample_id"] = np.concatenate(preds[3::4])
    data["label_start"] = torch.cat(labels[::2]).cpu().numpy()
    data["label_end"] = torch.cat(labels[1::2]).cpu().numpy()
    df = pd.DataFrame(data=data)

    # we sometimes have multiple predictions for one sample (= paragraph question pair)
    # because we split the paragraph into smaller passages
    # we want to check weather this sample belongs to is_impossible (all 0 start + end labels for all passages)
    # and compute metrics for the most likely prediction
    unique_sample_ids = df.sample_id.unique()
    em_all = []
    f1_all = []
    precision_all = []
    recall_all = []
    for uid in unique_sample_ids:
        group = df.loc[df.sample_id == uid,:]
        is_impossible = (np.sum(group.label_start) + np.sum(group.label_end)) == 0
        max_pred = group.loc[group.prob == np.max(group.prob)]
        if(max_pred.shape[0] > 1):
            max_pred = max_pred.loc[0,:] # hack away and just take first pred. Should rarely occur.
            logger.info(f"Multiple predictions having exactly the same probability value. "
                        f"Something might be wrong at sample ids: {str(max_pred.sample_id.values)}")
        if not is_impossible:
            # cover special case: for a doc splitted into multiple passages, there is one passage with a text label
            # but we have max_pred pointing to another passage that did not contain this answer (so the label for
            # this passage is also "no_answer")

            # TODO: add weighting of no answer predicitons vs answer predictions
            if(max_pred.pred_start.values[0] + max_pred.pred_end.values[0] == 0):
                em_all.append(0)
                precision_all.append(0)
                recall_all.append(0)
                f1_all.append(0)
                continue
        em,p,r,f1 = compute_qa_f1(pred_start=max_pred.pred_start.values[0],
                                 pred_end=max_pred.pred_end.values[0],
                                 label_start= max_pred.label_start.values[0],
                                 label_end= max_pred.label_end.values[0])
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