from __future__ import absolute_import, division, print_function

import logging
import torch
from tqdm import tqdm

from farm.utils import MLFlowLogger as MlLogger
from farm.eval import Evaluator

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
        data_silo,
        epochs,
        n_gpu,
        device,
        warmup_linear=0.1,
        evaluate_every=100,
        evaluator_dev=None,
        evaluator_test=None,
        fp16=False,
        grad_acc_steps=1,
    ):
        self.data_silo = data_silo
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.fp16 = fp16
        self.learning_rate = self.optimizer.get_lr()
        self.warmup_linear = warmup_linear
        self.global_step = 0
        self.data_loader_train = data_silo.get_data_loader("train")
        self.device = device
        self.log_params()

        # evaluator on dev set
        if evaluator_dev is None:
            evaluator_dev = Evaluator(
                data_loader=self.data_silo.get_data_loader("dev"),
                label_list=self.data_silo.processor.label_list,
                device=device,
                metrics=self.data_silo.processor.metrics,
            )
        self.evaluator_dev = evaluator_dev

        # evaluator on test set
        if evaluator_test is None:
            evaluator_test = Evaluator(
                data_loader=self.data_silo.get_data_loader("test"),
                label_list=self.data_silo.processor.label_list,
                device=device,
                metrics=self.data_silo.processor.metrics,
            )
        self.evaluator_test = evaluator_test

    def train(self, model):
        logger.info("***** Running training *****")
        model.train()
        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(
                tqdm(self.data_loader_train, desc=f"Train epoch {epoch}/{self.epochs}")
            ):

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward pass through model
                logits = model.forward(**batch)
                per_sample_loss = model.logits_to_loss(logits=logits, **batch)

                self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if self.global_step != 1 and (
                    self.global_step % self.evaluate_every == 1
                ):
                    result = self.evaluator_dev.eval(model)
                    self.evaluator_dev.log_results(result, "Val", self.global_step)

                self.global_step += 1

        result = self.evaluator_test.eval(model)
        self.evaluator_test.log_results(result, "Test", self.global_step)
        return model

    def backward_propagate(self, loss, step):
        loss = self.adjust_loss(loss)
        if self.global_step % 10 == 1:
            MlLogger.log_metrics(
                {"Train_loss_total": float(loss.detach().cpu().numpy())},
                step=self.global_step,
            )
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
                # MlLogger.write_metrics({"learning_rate": lr_this_step}, step=self.global_step)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def adjust_loss(self, loss):
        loss = loss.mean()
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss

    def log_params(self):
        params = {"epochs": self.epochs, "n_gpu": self.n_gpu, "device": self.device}
        MlLogger.log_params(params)
