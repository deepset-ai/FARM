from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import f1_score as seq_f1_score
import numpy as np


def simple_accuracy(preds, labels):
    # TODO: THIS HACKY TRY CATCH IS FOR GNAD
    try:
        preds = np.array(preds)
        labels = np.array(labels)
        correct = preds = labels
        return {
            "acc" :correct.mean()
        }
    except TypeError:
        return (preds == labels.numpy()).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def f1_macro(preds,labels):
    return {
        "f1_macro" : f1_score(y_true=labels,y_pred=preds,average="macro")
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(metric, preds, labels):
    assert len(preds) == len(labels)
    if metric == "mcc":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif metric == "acc":
        return {"acc": simple_accuracy(preds, labels)}
    elif metric == "acc_f1":
        return acc_and_f1(preds, labels)
    elif metric == "pear_spear":
        return pearson_and_spearman(preds, labels)
    elif metric == "seq_f1":
        return {"seq_f1": seq_f1_score(labels, preds)}
    elif metric == "f1_macro":
        return f1_macro(preds, labels)
    else:
        raise KeyError(metric)
