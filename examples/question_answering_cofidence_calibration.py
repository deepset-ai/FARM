import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from farm.evaluation.metrics import squad_f1_single
from farm.infer import QAInferencer


def _inference_f1(res):
    f1_scores = []

    for qapred in res:
        if qapred.prediction[0].answer == "no_answer":
            best_pred = (-1,-1)
        else:
            best_pred = (qapred.prediction[0].offset_answer_start, qapred.prediction[0].offset_answer_end)
        if len(qapred.ground_truth_answer) == 0:
            labels = [(-1,-1)]
        else:
            labels = []
            for gt in qapred.ground_truth_answer:
                labels.append((gt["answer_start"],gt["answer_start"]+len(gt["text"])))
        best_f1 = max([_qa_f1_single(best_pred, label) for label in labels])
        f1_scores.append(best_f1)
    return f1_scores

def _qa_f1_single(pred, label):
    label_start, label_end = label
    pred_start,pred_end = pred


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

def extract_preds_labels(datafile, model = "deepset/minilm-uncased-squad2"):
    model = QAInferencer.load(
                model_name_or_path=model,
                task_type="question_answering",
                batch_size=80,
                gpu=True
    )

    res = model.inference_from_file(datafile,return_json=False)
    return res


def calibrate_confidence(res):
    # extract weather gold label was text or no_answer
    text_answer = []
    for qapred in res:
        if len(qapred.ground_truth_answer) == 0:
            text_answer.append(0)
        else:
            text_answer.append(1)

    text_answer = np.array(text_answer)

    # extract features and f1 scores
    features = []
    f1_scores = np.array(_inference_f1(res))
    for qapred in res:
        feat = qapred.prediction[0].conf_scores.copy()  # conf_scores =   [start_logits[0].item(), end_logits[0].item(), score_start, score_end]
        feat.append((feat[0]+feat[1]) > (feat[2]+feat[3]))
        feat = np.array(feat)
        features.append(feat)
    features = np.array(features)

    for THRESHOLD in [0.8,0.9,1]:
        # create hit or miss labels
        conf_labels = f1_scores >= THRESHOLD

        # potentially separate answer vs no_answer TODO understand why no_answer confidence can be exactly determined
        #features = features[text_answer == 1]
        #conf_labels = conf_labels[text_answer == 1]

        #train test split
        X_train, X_test, y_train, y_test = train_test_split(features, conf_labels, test_size = 0.33)

        # apply logistic regression
        # Fit classifier with out-of-bag estimates
        params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
                  'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        # print scores
        print(f"Statistics on f1 score bigger than: {THRESHOLD}")
        print(f"test accuracy score {clf.score(X_test, y_test)} - majority vote accuracy: {sum(y_test)/y_test.shape[0]}")
        preds = clf.predict(X_test)
        print(classification_report(y_true=y_test, y_pred=preds))

        print(f"Fteaure importance: {clf.feature_importances_}")

if __name__ == "__main__":
    datafile = "../data/squad20/dev-v2.0.json"
    model = "deepset/minilm-uncased-squad2"
    #res = extract_preds_labels(datafile=datafile, model=model)

    #pickle.dump(res,open("../data/squad20/res.pkl","wb"))
    res = pickle.load(open("../data/squad20/res.pkl","rb"))

    calibrate_confidence(res)