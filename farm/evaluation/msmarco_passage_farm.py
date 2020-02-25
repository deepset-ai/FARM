from farm.evaluation.msmarco_passage_official import compute_metrics_from_files
import os
import pandas as pd


def msmarco_evaluation(preds_file, dev_file, qrels_file, output_file):
    """
    Performs official msmarco passage ranking evaluation (https://github.com/microsoft/MSMARCO-Passage-Ranking)
    on a file containing the is_relevent prediction scores. It will convert the input file (qid, pid, score)
    into the format expected by the official eval function (compute_metrics_from_files)

    :param predictions_filename: File where each line is the is_relevant prediction score
    :param dev_filename: File in format qid, query, pid, passage, label
    :param qrels_filename: File in the format qid, pid when is_relevant=1
    :param output_file: File to write to in format qid, pid, rank

    :return:
    """

    # Initialize files
    preds_scores = [float(l) for l in open(preds_file)]
    dev_lines = [l for i,l in enumerate(open(dev_file)) if i != 0]
    output = open(output_file, "w")

    # Populate a dict with all qid/pid/score triples
    results = dict()
    for i, (score, line) in enumerate(zip(preds_scores, dev_lines)):
        if i == 0:
            continue
        qid, _, pid, _, _ = line.split("\t")
        if qid not in results:
            results[qid] = []
        results[qid].append((pid, score))

    # ##########
    # ### NOTE: This block is to generate a view that is interpretable when debugging
    # ##########
    # interpretable = dict()
    # for i, (score, line) in enumerate(zip(preds_scores, dev_lines)):
    #     if i == 0:
    #         continue
    #     _, query, _, passage, label = line.split("\t")
    #     if query not in interpretable:
    #         interpretable[query] = []
    #     interpretable[query].append((passage, score, label[:-1]))
    # for query in interpretable:
    #     sorted_scores = sorted(interpretable[query], key= lambda x: x[1], reverse=True)[:10]
    #     results[query] = sorted_scores
    # relevant = []
    # for query in interpretable:
    #     for (passage, score, label) in interpretable[query]:
    #         if label == "1":
    #             relevant.append((passage, score))
    # rel_scores = [x[1] for x in relevant]
    # irrelevant = []
    # for query in interpretable:
    #     for (passage, score, label) in interpretable[query]:
    #         if label == "0":
    #             irrelevant.append((passage, score))
    # irrel_scores = [x[1] for x in irrelevant]
    # print()

    # Sort by scores and take top 10
    for qid in list(results):
        sorted_scores = sorted(results[qid], key= lambda x: x[1], reverse=True)[:10]
        results[qid] = [(pid, i+1) for i, (pid, _)  in enumerate(sorted_scores)]

    # Write to file
    for qid in list(results):
        for (pid, rank) in results[qid]:
            output.write(f"{qid}\t{pid}\t{rank}\n")
    output.close()

    curr_qids = list(results)
    df = pd.read_csv(qrels_file, sep="\t", header=None)
    df = df.loc[df[0].isin(curr_qids)]
    df.to_csv("tmp", sep="\t", header=None, index=None)

    path_to_reference = "tmp"
    path_to_candidate = output_file
    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print('#####################')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('#####################')
    os.remove(path_to_reference)



