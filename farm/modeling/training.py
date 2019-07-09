from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import torch
from seqeval.metrics import classification_report as token_classification_report
from sklearn.metrics import classification_report
from tqdm import tqdm, trange

from farm.metrics import compute_metrics


logger = logging.getLogger(__name__)


class WrappedDataParallel(torch.nn.DataParallel):
    """
    Hack to get attributes of underlying class in parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see: https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


try:
    from apex.parallel import DistributedDataParallel as DDP

    class WrappedDDP(DDP):
        """
        Hack to get attributes of underlying class in distributed mode. Same as in WrappedDataParallel above.
        Even when using distributed on a single machine with multiple GPUs, apex can speed up training significantly.
        Distributed code must be launched with "python -m torch.distributed.launch --nproc_per_node=1 run_script.py"
        """

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)


except ImportError:
    logger.warn(
        "Apex not installed. If you use distributed training with local rank != -1 apex must be installed."
    )


class Trainer:
    def __init__(
        self,
        optimizer,
        data_bunch,
        evaluator_dev,
        epochs,
        n_gpu,
        grad_acc_steps,
        fp16,
        learning_rate,
        warmup_linear,
        device,
        evaluate_every=100,
    ):
        self.data_bunch = data_bunch
        self.evaluator_dev = evaluator_dev
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.warmup_linear = warmup_linear
        self.global_step = 0
        self.data_loader_train = data_bunch.get_data_loader("train")
        self.device = device

    def train(self, model):
        logger.info("***** Running training *****")
        model.train()
        for _ in trange(self.epochs, desc="Epoch"):
            for step, batch in enumerate(
                tqdm(self.data_loader_train, desc="Iteration")
            ):

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward pass through model
                logits = model.forward(**batch)
                loss = model.logits_to_loss(logits=logits, **batch)

                self.backward_propagate(loss, step)

                # Perform evaluation
                if self.global_step % self.evaluate_every == 1:
                    result = self.evaluator_dev.eval(model)
                    self.evaluator_dev.print_results(result, "Dev", self.global_step)
                    # # Log to mlflow
                    # #TODO make it optional
                    # metrics = {f"dev {metric_name}": metric_val for metric_name, metric_val in result.items()}
                    # MLFlowLogger.write_metrics(metrics, step=self.global_step)

                self.global_step += 1
        return model

    def backward_propagate(self, loss, step):
        loss = self.adjust_loss(loss)
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % self.grad_acc_steps == 0:
            if self.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = self.learning_rate * self.warmup_linear.get_lr(
                    self.global_step, self.warmup_proportion
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()

    def adjust_loss(self, loss):
        loss = loss.mean()
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss


class Evaluator:
    def __init__(
        self, data_loader, label_list, device, metrics, classification_report=True
    ):

        self.data_loader = data_loader
        self.label_map = {i: label for i, label in enumerate(label_list)}
        self.device = device

        # Where should metric be defined? When dataset loaded? In config?
        self.metrics = metrics
        self.classification_report = classification_report

    def eval(self, model):
        model.eval()

        # init empty lists per prediction head
        loss_all = [[] for _ in model.prediction_heads]
        logits_all = [[] for _ in model.prediction_heads]
        preds_all = [[] for _ in model.prediction_heads]
        label_all = [[] for _ in model.prediction_heads]

        for step, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():

                logits = model.forward(**batch)
                # todo logits_to_loss should be a single, overloaded function
                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)

                preds = model.logits_to_preds(
                    logits=logits, label_map=self.label_map, **batch
                )

                labels = model.prepare_labels(label_map=self.label_map, **batch)

            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += list(to_numpy(losses_per_head[head_num]))
                logits_all[head_num] += list(to_numpy(logits[head_num]))
                preds_all[head_num] += list(to_numpy(preds[head_num]))
                label_all[head_num] += list(to_numpy(labels[head_num]))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            result = {"loss_eval": np.mean(loss_all[head_num])}
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
                else:
                    raise NotImplementedError
                result["report"] = report_fn(
                    label_all[head_num], preds_all[head_num], digits=4
                )
            all_results.append(result)

        return all_results

    @staticmethod
    def print_results(results, dataset_name, steps):
        logger.info(
            "***** {} Eval Results After Steps: {} *****".format(dataset_name, steps)
        )
        for num, head in enumerate(results):
            logger.info("\n _________ Prediction Head {} _________".format(num))
            for key, value in head.items():
                logger.info("{}".format(key))
                logger.info("{}".format(value))


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container
