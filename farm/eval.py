from tqdm import tqdm
import torch
import numbers
import logging
import numpy as np
from seqeval.metrics import classification_report as token_classification_report
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from farm.metrics import compute_metrics
from farm.utils import to_numpy
from farm.utils import MLFlowLogger as MlLogger
from farm.modeling.adaptive_model import AdaptiveModel

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles evaluation of a given model over a specified dataset."""

    def __init__(
        self, data_loader, label_maps, device, metrics, classification_report=True
    ):
        """
        :param data_loader: The PyTorch DataLoader that will return batches of data from the evaluation dataset
        :type data_loader: DataLoader
        :param label_maps:
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
        :param metrics: The list of metrics which need to be computed, one for each prediction head.
        :param metrics: list
        :param classification_report: Whether a report on the classification performance should be generated.
        :type classification_report: bool
        """

        self.data_loader = data_loader
        self.label_maps = label_maps

        self.device = device

        # Where should metric be defined? When dataset loaded? In config?
        self.metrics = metrics
        self.classification_report = classification_report

    def eval(self, model):
        """
        Performs evaluation on a given model.

        :param model: The model on which to perform evaluation
        :type model: AdaptiveModel
        :return all_results: A list of dictionaries, one for each prediction head. Each dictionary contains the metrics
                             and reports generated during evaluation.
        :rtype all_results: list of dicts
        """
        model.eval()

        # init empty lists per prediction head
        loss_all = [[] for _ in model.prediction_heads]
        logits_all = [[] for _ in model.prediction_heads]
        preds_all = [[] for _ in model.prediction_heads]
        label_all = [[] for _ in model.prediction_heads]

        for step, batch in enumerate(
            tqdm(self.data_loader, desc="Evaluating", mininterval=10)
        ):
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():

                logits = model.forward(**batch)
                # TODO logits_to_loss should be a single, overloaded function
                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)

                preds = model.logits_to_preds(
                    logits=logits, label_maps=self.label_maps, **batch
                )

                labels = model.prepare_labels(label_maps=self.label_maps, **batch)

            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += list(to_numpy(losses_per_head[head_num]))
                logits_all[head_num] += list(to_numpy(logits[head_num]))
                preds_all[head_num] += list(to_numpy(preds[head_num]))
                label_all[head_num] += list(to_numpy(labels[head_num]))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            result = {"loss": np.mean(loss_all[head_num])}
            result.update(
                compute_metrics(
                    self.metrics[head_num], preds_all[head_num], label_all[head_num]
                )
            )

            # Select type of report depending on prediction head output type
            if self.classification_report:
                if head.ph_output_type == "per_token":
                    report_fn = token_classification_report
                elif head.ph_output_type == "per_sequence":
                    report_fn = classification_report
                elif head.ph_output_type == "per_token_squad":
                    report_fn = lambda *args, **kwargs: "not Implemented"
                else:
                    raise NotImplementedError
                result["report"] = report_fn(
                    label_all[head_num], preds_all[head_num], digits=4
                )
            all_results.append(result)

        return all_results

    @staticmethod
    def log_results(results, dataset_name, steps, logging=True, print=True):
        logger.info(
            "\n***** Evaluation Results on {} data after {} steps *****".format(
                dataset_name, steps
            )
        )
        for head_num, head in enumerate(results):
            logger.info("\n _________ Prediction Head {} _________".format(head_num))
            for metric_name, metric_val in head.items():
                # log with ML framework (e.g. Mlflow)
                if logging:
                    if isinstance(metric_val, numbers.Number):
                        MlLogger.log_metrics(
                            metrics={
                                f"{dataset_name}_{metric_name}_head{head_num}": metric_val
                            },
                            step=steps,
                        )
                # print via standard python logger
                if print:
                    if metric_name == "report":
                        if len(metric_val) > 8000:
                            metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                        logger.info("{}: \n {}".format(metric_name, metric_val))
                    else:
                        logger.info("{}: {}".format(metric_name, metric_val))
