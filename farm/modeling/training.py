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
        evaluator_test,
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
        self.evaluator_test = evaluator_test
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
                # input_ids, padding_mask, segment_ids, label_ids, initial_mask = batch

                # Forward pass through model
                logits = model.forward(**batch)
                loss = model.logits_to_loss(logits=logits, **batch)

                self.backward_propagate(loss, step)

                # Perform evaluation
                if self.global_step % self.evaluate_every == 1:
                    result = self.evaluator_dev.eval(model)
                    self.print_dev(result, self.global_step)
                    # # Log to mlflow
                    # #TODO make it optional
                    # metrics = {f"dev {metric_name}": metric_val for metric_name, metric_val in result.items()}
                    # MLFlowLogger.write_metrics(metrics, step=self.global_step)

                self.global_step += 1
        return model

    def evaluate_on_test(self, model):
        result = self.evaluator_test.eval(model)
        logger.info("***** Test Eval Results *****")
        logger.info(result["report"])

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

    @staticmethod
    def print_dev(result, step):
        logger.info("***** Dev Eval Results After Steps: {} *****".format(step))
        logger.info(result["report"])


class Evaluator:
    def __init__(self, data_loader, label_list, device, metric, ph_output_type):

        self.data_loader = data_loader
        self.label_map = {i: label for i, label in enumerate(label_list)}

        # These will contain the per sample loss, logits, preds and Y
        self.loss_all = []
        self.logits_all = []
        self.preds_all = []
        self.Y_all = []

        self.device = device
        # Where should metric be defined? When dataset loaded? In config?
        self.metric = metric

        # Turn classification_report into an argument of init
        if ph_output_type == "per_token":
            self.classification_report = token_classification_report
        elif ph_output_type == "per_sequence":
            self.classification_report = classification_report
        else:
            raise NotImplementedError

    def eval(self, model):
        model.eval()

        for step, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
            batch = {key: batch[key].to(self.device) for key in batch}
            # input_ids, padding_mask, segment_ids, label_ids, initial_mask = batch

            with torch.no_grad():

                logits = model.forward(**batch)
                loss = model.logits_to_loss(logits=logits, **batch)

                # Todo BC: I don't like that Y is returned here but this is the best I have right now
                Y, preds = model.logits_to_preds(
                    logits=logits, label_map=self.label_map, **batch
                )
            # TODO this needs to be resolved
            label_ids = batch["label_ids"]
            if Y is not None:
                label_ids = Y

            self.loss_all += list(to_numpy(loss))
            self.logits_all += list(to_numpy(logits))
            self.preds_all += list(to_numpy(preds))
            self.Y_all += list(to_numpy(label_ids))

        loss_eval = np.mean(self.loss_all)
        result = {"loss_eval": loss_eval}
        result[self.metric] = compute_metrics(self.metric, self.preds_all, self.Y_all)
        result["report"] = self.classification_report(
            self.Y_all, self.preds_all, digits=4
        )

        self.reset_state()
        return result

    def reset_state(self):
        self.loss_all = []
        self.logits_all = []
        self.preds_all = []
        self.Y_all = []


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container
