from tqdm import tqdm
import torch
import numbers
import logging
import numpy as np
from seqeval.metrics import classification_report as token_classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from farm.metrics import compute_metrics
from farm.utils import to_numpy, format_log
from farm.utils import MLFlowLogger as MlLogger
from farm.modeling.adaptive_model import AdaptiveModel
from farm.visual.ascii.images import BUSH_SEP

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles evaluation of a given model over a specified dataset."""

    def __init__(
        self, data_loader, tasks, device, report=True
    ):
        """
        :param data_loader: The PyTorch DataLoader that will return batches of data from the evaluation dataset
        :type data_loader: DataLoader
        :param label_maps:
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
        :param metrics: The list of metrics which need to be computed, one for each prediction head.
        :param metrics: list
        :param report: Whether an eval report should be generated (e.g. classification report per class).
        :type report: bool
        """

        self.data_loader = data_loader
        self.tasks = tasks
        self.device = device
        self.report = report

    def eval(self, model, return_preds_and_labels=False):
        """
        Performs evaluation on a given model.

        :param model: The model on which to perform evaluation
        :type model: AdaptiveModel
        :param return_preds_and_labels: Whether to add preds and labels in the returned dicts of the
        :type return_preds_and_labels: bool
        :return all_results: A list of dictionaries, one for each prediction head. Each dictionary contains the metrics
                             and reports generated during evaluation.
        :rtype all_results: list of dicts
        """
        model.eval()

        # init empty lists per prediction head
        loss_all = [0 for _ in model.prediction_heads]
        preds_all = [[] for _ in model.prediction_heads]
        label_all = [[] for _ in model.prediction_heads]
        ids_all = [[] for _ in model.prediction_heads]
        passage_start_t_all = [[] for _ in model.prediction_heads]

        for step, batch in enumerate(
            tqdm(self.data_loader, desc="Evaluating", mininterval=10)
        ):
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():

                logits = model.forward(**batch)
                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)
                preds = model.logits_to_preds(logits=logits, **batch)
                labels = model.prepare_labels(**batch)

            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += np.sum(to_numpy(losses_per_head[head_num]))
                preds_all[head_num] += list(to_numpy(preds[head_num]))
                label_all[head_num] += list(to_numpy(labels[head_num]))
                if head.model_type == "span_classification":
                    ids_all[head_num] += list(to_numpy(batch["id"]))
                    passage_start_t_all[head_num] += list(to_numpy(batch["passage_start_t"]))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            if head.model_type == "multilabel_text_classification":
                # converting from string preds back to multi-hot encoding
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer(classes=head.label_list)
                # TODO check why .fit() should be called on predictions, rather than on labels
                preds_all[head_num] = mlb.fit_transform(preds_all[head_num])
                label_all[head_num] = mlb.transform(label_all[head_num])
            if hasattr(head, 'aggregate_preds'):
                preds_all[head_num], label_all[head_num] = head.aggregate_preds(preds=preds_all[head_num],
                                                                          labels=label_all[head_num],
                                                                          passage_start_t=passage_start_t_all[head_num],
                                                                          ids=ids_all[head_num])

            result = {"loss": loss_all[head_num] / len(self.data_loader.dataset),
                      "task_name": head.task_name}
            result.update(
                compute_metrics(metric=head.metric, preds=preds_all[head_num], labels=label_all[head_num]
                )
            )

            # Select type of report depending on prediction head output type
            if self.report:
                if head.ph_output_type == "per_token":
                    report_fn = token_classification_report
                elif head.ph_output_type == "per_sequence":
                    report_fn = classification_report
                elif head.ph_output_type == "per_token_squad":
                    report_fn = lambda *args, **kwargs: "not Implemented"
                elif head.ph_output_type == "per_sequence_continuous":
                    report_fn = r2_score
                else:
                    raise NotImplementedError

                # CHANGE PARAMETERS, not all report_fn accept digits
                if head.ph_output_type in ["per_sequence_continuous","per_token"]:
                    result["report"] = report_fn(
                        label_all[head_num], preds_all[head_num]
                    )
                else:
                    # supply labels as all possible combination because if ground truth labels do not cover
                    # all values in label_list (maybe dev set is small), the report will break
                    if head.model_type == "multilabel_text_classification":
                        # For multilabel classification, we don't eval with string labels here, but with multihot vectors.
                        # Therefore we need to supply all possible label ids instead of label values.
                        all_possible_labels = list(range(len(head.label_list)))
                    else:
                        all_possible_labels = head.label_list

                    result["report"] = report_fn(
                        label_all[head_num],
                        preds_all[head_num],
                        digits=4,
                        labels=all_possible_labels,
                        target_names=head.label_list)

            if return_preds_and_labels:
                result["preds"] = preds_all[head_num]
                result["labels"] = label_all[head_num]

            all_results.append(result)

        return all_results

    @staticmethod
    def log_results(results, dataset_name, steps, logging=True, print=True, num_fold=None):
        # Print a header
        header = "\n\n"
        header += BUSH_SEP + "\n"
        header += "***************************************************\n"
        if num_fold:
            header += f"***** EVALUATION | FOLD: {num_fold} | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        else:
            header += f"***** EVALUATION | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        header += "***************************************************\n"
        header += BUSH_SEP + "\n"
        logger.info(header)

        for head_num, head in enumerate(results):
            logger.info("\n _________ {} _________".format(head['task_name']))
            for metric_name, metric_val in head.items():
                # log with ML framework (e.g. Mlflow)
                if logging:
                    if not metric_name in ["preds","labels"] and not metric_name.startswith("_"):
                        if isinstance(metric_val, numbers.Number):
                            MlLogger.log_metrics(
                                metrics={
                                    f"{dataset_name}_{metric_name}_{head['task_name']}": metric_val
                                },
                                step=steps,
                            )
                # print via standard python logger
                if print:
                    if metric_name == "report":
                        if isinstance(metric_val, str) and len(metric_val) > 8000:
                            metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                        logger.info("{}: \n {}".format(metric_name, metric_val))
                    else:
                        if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                            logger.info("{}: {}".format(metric_name, metric_val))
