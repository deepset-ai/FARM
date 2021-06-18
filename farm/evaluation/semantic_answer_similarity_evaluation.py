from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from farm.evaluation.metrics import top_n_accuracy


def semantic_answer_similarity(result, sts_model_path_or_string="paraphrase-xlm-r-multilingual-v1", debug=False):
    """
    Computes BERT based similarity of prediction to gold labels.
    Returns per QA pair a) the similarity of the most likely prediction to all available gold labels
                        b) the highest similarity of all predictions to gold labels
                        c) the original input populated with semantic answer scores per prediction

    :param result: Output of QAinferencer.inference_from_file when setting return_json=False. Contains predictions
                   and annotation information.
    :type result: List[QAPred]
    :param sts_model_path_or_string: SentenceTransformers semantic textual similarity model, should be path or string
                                     pointing to downloadable models.
                                     Defaults to multilingual model = "paraphrase-xlm-r-multilingual-v1"
                                     For more speed and English only, use "paraphrase-MiniLM-L6-v2"
    :type str
    :param debug: Collect additional info for interesting cases for debugging purposes (default False).
                  Interesting cases are predictions that do not overlap with any of the gold labels (topnNAcc == 0).

    :type debug: bool

    Returns the average of the semantically evaluated best, and Top N predictions as well as the List of
    QApreds (result) with the semantic eval score added to each prediction in its "meta" field.
    :return best_pred_similarity, all_preds_highest_similarity, result, debug_info
    """
    text_preds, text_labels = _extract_QA_result(result)
    assert len(text_preds) == len(text_labels)

    model = SentenceTransformer(sts_model_path_or_string)
    # Flatten predictions and labels into one list to improve encoding speed through proper batching
    lengths = []
    all_texts = []
    for p, l in zip(text_preds, text_labels):
        # TODO potentially exclude (near) exact matches from computations
        all_texts.extend(p)
        all_texts.extend(l)
        lengths.append((len(p), len(l)))
    embeddings = model.encode(all_texts)

    # Compute similarities
    top1_sim = []
    topn_sim = []
    debug_info = []

    current_position = 0
    for i, (len_p, len_l) in enumerate(lengths):
        pred_embeddings = embeddings[current_position:current_position + len_p, :]
        current_position += len_p
        label_embeddings = embeddings[current_position:current_position + len_l, :]
        current_position += len_l
        sims = cosine_similarity(pred_embeddings, label_embeddings)
        top1_sim.append(np.max(sims[0, :]))
        topn_sim.append(np.max(sims))
        for j,p in enumerate(result[i].prediction):
            p.meta = {"semantic_answer_score":np.max(sims[j, :])}

        # Add debug information
        if debug:
            current_info = _add_debug_info(result=result, i=i, sims=sims)
            if current_info is not None:
                debug_info.append(current_info)

    return np.mean(top1_sim), np.mean(topn_sim), result, debug_info


# helper fct to extract labels and predictions in text form
def _extract_QA_result(result):
    text_preds = []
    text_labels = []
    for r in result:
        current_labels = []
        current_preds = []
        if len(r.ground_truth_answer) == 0:
            current_labels.append("no_answer")
        else:
            for a in r.ground_truth_answer:
                current_labels.append(a["text"])
        text_labels.append(current_labels)

        for p in r.prediction:
            current_preds.append(p.answer)
        text_preds.append(current_preds)

    return text_preds, text_labels


# helper fct to add debug info
def _add_debug_info(result,i,sims):
    current_preds = [[result[i].prediction]]
    current_labels = [
        [(x["answer_start"], x["answer_start"] + len(x["text"])) for x in result[i].ground_truth_answer]]
    # Exclude no answer labels from debug info
    if len(result[i].ground_truth_answer) == 0:
        return None
    # Only add debug info if there is no overlap between predictions and labels
    elif top_n_accuracy(preds=current_preds, labels=current_labels) == 0:
        current_info = {}
        current_info["question"] = result[i].question
        current_info["top1_sim"] = np.max(sims[0, :])
        current_info["top1_label"] = result[i].ground_truth_answer[np.argmax(sims[0, :])]["text"]
        current_info["top1_pred"] = result[i].prediction[0].answer

        idx = np.unravel_index(sims.argmax(), sims.shape)
        current_info["topn_sim"] = np.max(sims)
        current_info["topn_label"] = result[i].ground_truth_answer[idx[1]]["text"]
        current_info["topn_pred"] = result[i].prediction[idx[0]].answer
        return current_info
    else:
        return None
