import pytest
import math
from farm.evaluation.metrics import compute_metrics


def test_compute_metrics_basic():
    # check we get some exception, may not always be the AssertionError we get now
    with pytest.raises(Exception):
        compute_metrics("acc", ["x"] * 10, [""] * 11)
    ret = compute_metrics("acc", [], [])
    assert isinstance(ret, dict)
    assert "acc" in ret
    assert math.isnan(ret["acc"])
    with pytest.raises(Exception):
        compute_metrics("asdfasdf", ["a"], ["b"])
    ls = (["a"] * 5)
    ls.extend(["b"] * 5)
    ps = ["a"] * 10
    ret = compute_metrics("acc", ps, ls)
    assert ret["acc"] == 0.5
    ret = compute_metrics("acc", ls, ps)
    assert ret["acc"] == 0.5
    ret = compute_metrics("f1_macro", ps, ls)
    assert ret["f1_macro"] == 1/3
    ret = compute_metrics("f1_macro", ls, ps)
    assert ret["f1_macro"] == 1 / 3
    ret = compute_metrics(["f1_macro", "acc"], ps, ls)
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert "acc" in ret
    assert "f1_macro" in ret
    assert ret["f1_macro"] == 1/3
    assert ret["acc"] == 0.5
    ret = compute_metrics(["f1_macro", "acc", "acc"], ps, ls)
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert "acc" in ret
    assert "f1_macro" in ret
    assert ret["f1_macro"] == 1/3
    assert ret["acc"] == 0.5
    ret = compute_metrics(["f1_macro", ["acc"]], ps, ls)
    assert isinstance(ret, dict)
    assert len(ret) == 2
    assert "acc" in ret
    assert "f1_macro" in ret
    assert ret["f1_macro"] == 1/3
    assert ret["acc"] == 0.5

from farm.infer import Inferencer
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from farm.evaluation.metrics import top_n_accuracy

def test_QA_answer_similarity():#bert_base_squad2):

    #model = Inferencer.load(model_name_or_path="deepset/gelectra-base-germanquad",task_type="question_answering",batch_size=32)
    #result = model.inference_from_file(file="samples/qa/germanquad_test.json",return_json=False)

    #pickle.dump(result,open("samples/qa/resfull.pkl","wb"))
    result = pickle.load(open("samples/qa/resfull.pkl","rb"))
    text_preds, text_labels = _extract_QA_result(result)
    sims = answer_similarity(text_preds, text_labels,result=result)
    muh=1

    assert True


def _extract_QA_result(result):
    text_preds = []
    text_labels = []
    for r in result:
        current_labels = []
        current_preds = []
        for a in r.ground_truth_answer:
            current_labels.append(a["text"])
        text_labels.append(current_labels)

        for p in r.prediction:
            current_preds.append(p.answer)
        text_preds.append(current_preds)

    return text_preds, text_labels



def answer_similarity(text_preds, text_labels, result, debug=True):
    """
    Computes BERT based similarity of prediction to gold labels.
    Returns per QA pair a) the similarity of the most likely prediction to all available gold labels
                        b) the highest similarity of all predictions to gold labels

    :param text_preds:
    :type text_preds:
    :param text_labels:
    :type text_labels:

    :return best_pred_similarity, all_preds_highest_similarity
    """
    # TODO dont use cross en de roberta
    ######model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

    #model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    # flatten predictions and labels into one list to better batch the encoding
    lengths = []
    all_texts = []
    for p,l in zip(text_preds,text_labels):
        all_texts.extend(p)
        all_texts.extend(l)
        lengths.append((len(p),len(l)))
    #embeddings = model.encode(all_texts)
    #pickle.dump(embeddings,open("samples/qa/resfullemb.pkl","wb"))
    embeddings = pickle.load(open("samples/qa/resfullemb.pkl", "rb"))

    # compute similarities
    top1_sim = []
    topn_sim = []
    debug_info = []

    current_position = 0
    for i,(len_p,len_l) in enumerate(lengths):
        pred_embeddings = embeddings[current_position:current_position+len_p,:]
        current_position += len_p
        label_embeddings = embeddings[current_position:current_position+len_l,:]
        current_position += len_l
        sims = cosine_similarity(pred_embeddings,label_embeddings)
        top1_sim.append(np.max(sims[0,:]))
        topn_sim.append(np.max(sims))
        if debug:
            # TODO handle NO answer labels
            current_preds = [[result[i].prediction]]
            current_labels = [[(x["answer_start"],x["answer_start"]+len(x["text"])) for x in result[i].ground_truth_answer]]
            if top_n_accuracy(preds=current_preds,labels=current_labels) == 0:
                current_info = {}
                current_info["question"] = result[i].question
                current_info["top1_sim"] = np.max(sims[0,:])
                current_info["top1_label"] = result[i].ground_truth_answer[np.argmax(sims[0,:])]["text"]
                current_info["top1_pred"] = result[i].prediction[0].answer

                idx = np.unravel_index(sims.argmax(),sims.shape)
                current_info["topn_sim"] = np.max(sims)
                current_info["topn_label"] = result[i].ground_truth_answer[idx[1]]["text"]
                current_info["topn_pred"] = result[i].prediction[idx[0]].answer
                debug_info.append(current_info)
        pass


    pass

if __name__ == "__main__":
    test_QA_answer_similarity()