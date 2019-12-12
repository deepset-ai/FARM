from __future__ import absolute_import, division, print_function

import logging
import torch
from tqdm import tqdm

from farm.utils import MLFlowLogger as MlLogger
from farm.eval import Evaluator
from farm.data_handler.data_silo import DataSilo
from farm.visual.ascii.images import GROWING_TREE
from farm.modeling.adaptive_model import AdaptiveModel

logger = logging.getLogger(__name__)


class WrappedDataParallel(torch.nn.DataParallel):
    """
    A way of adapting attributes of underlying class to parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

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
        A way of adapting attributes of underlying class to distributed mode. Same as in WrappedDataParallel above.
        Even when using distributed on a single machine with multiple GPUs, apex can speed up training significantly.
        Distributed code must be launched with "python -m torch.distributed.launch --nproc_per_node=1 run_script.py"
        """

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)


except ImportError:
    logger.warning(
        "Apex not installed. If you use distributed training with local rank != -1 apex must be installed."
    )


class EarlyStopping:

    def __init__(
            self,
            metric="loss",
            save_dir=None,
            mode="min",
            patience=0,
            min_delta=0.001,
            min_evals=0,
    ):
        """
        Can be used to control early stopping with a Trainer class. Any object can be used instead which
        implements the method check_stopping and and provides the attribute save_dir
        :param save_dir: the directory where to save the final best model, if None, no saving.
        :param metric: name of dev set metric to monitor (default: loss) to get extracted from the 0th head or
        a function that extracts a value from the trainer dev evaluation result.
        NOTE: this is different from the metric to get specified for the processor which defines how
        to calculate one or more evaluation matric values from prediction/target sets, while this
        specifies the name of one particular such metric value or a method to calculate that value
        from the result returned from a processor metric.
        :param mode: "min" or "max"
        :param patience: how many evaluations to wait after the best evaluation to stop
        :param min_delta: minimum difference to a previous best value to count as an improvement.
        :param min_evals: minimum number of evaluations to wait before using eval value

        """
        self.metric = metric
        self.save_dir = save_dir
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.min_evals = min_evals
        self.eval_values = []  # for more complex modes
        self.n_since_best = None
        if mode == "min":
            self.best_so_far = 1.0E99
        elif mode == "max":
            self.best_so_far = -1.0E99
        else:
            raise Exception("Mode must be 'min' or 'max'")

    def check_stopping(self, eval_result):
        """
        Provide the evaluation value for the current evaluation. Returns true if stopping should occur.
        This will save the model, if necessary.
        :param eval: the current evaluation result
        :return: a tuple (stopprocessing, savemodel, evalvalue) indicating if processing should be stopped
        and if the current model should get saved and the evaluation value used.
        """
        if isinstance(self.metric, str):
            eval_value = eval_result[0][self.metric]
        else:
            eval_value = self.metric(eval_result)
        self.eval_values.append(float(eval_value))
        stopprocessing, savemodel = False, False
        if len(self.eval_values) <= self.min_evals:
            return stopprocessing, savemodel
        if self.mode == "min":
            delta = self.best_so_far - eval_value
        else:
            delta = eval_value - self.best_so_far
        if delta > self.min_delta:
            self.best_so_far = eval_value
            self.n_since_best = 0
            if self.save_dir:
                savemodel = True
        else:
            self.n_since_best += 1
        if self.n_since_best > self.patience:
            stopprocessing = True
        return stopprocessing, savemodel, eval_value


class Trainer:
    """Handles the main model training procedure. This includes performing evaluation on the dev set at regular
    intervals during training as well as evaluation on the test set at the end of training."""

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
        local_rank=-1,
        early_stopping=None,
    ):
        """
        :param optimizer: An optimizer object that determines the learning strategy to be used during training
        :param data_silo: A DataSilo object that will contain the train, dev and test datasets as PyTorch DataLoaders
        :type data_silo: DataSilo
        :param epochs: How many times the training procedure will loop through the train dataset
        :type epochs: int
        :param n_gpu: The number of gpus available for training and evaluation.
        :type n_gpu: int
        :param device: The device on which the train, dev and test tensors should be hosted. Choose from "cpu" and "cuda".
        :param warmup_linear: TODO
        :param evaluate_every: Perform dev set evaluation after this many steps of training.
        :type evaluate_every: int
        :param evaluator_dev: The dev set Evaluator object.
        :type evaluator_dev: Evaluator
        :param evaluator_test: The test set Evaluator object.
        :type evaluator_test: Evaluator
        :param fp16: Whether to use floating point 16 mode.
        :type fp16: bool
        :param grad_acc_steps: TODO
        :type grad_acc_steps: int
        :param local_rank: TODO
        :type local_rank: int
        :param early_stopping: an initialized EarlyStopping object to control early stopping and
        saving of best models.
        :type EarlyStopping
        """
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
        self.local_rank = local_rank
        self.log_params()
        self.early_stopping = early_stopping

        # evaluator on dev set
        if evaluator_dev is None and self.data_silo.get_data_loader("dev"):
            evaluator_dev = Evaluator(
                data_loader=self.data_silo.get_data_loader("dev"),
                tasks=self.data_silo.processor.tasks,
                device=device,
            )
        self.evaluator_dev = evaluator_dev

        # evaluator on test set
        if evaluator_test is None and self.data_silo.get_data_loader("test"):
            evaluator_test = Evaluator(
                data_loader=self.data_silo.get_data_loader("test"),
                tasks=self.data_silo.processor.tasks,
                device=device
            )
        self.evaluator_test = evaluator_test

    def train(self, model):
        """ Perform the training procedure. """

        # connect the prediction heads with the right output from processor
        model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Check that the tokenizer fits the language model
        model.verify_vocab_size(vocab_size=len(self.data_silo.processor.tokenizer))

        logger.info(f"\n {GROWING_TREE}")
        model.train()
        # multi GPU + distributed settings
        if self.fp16:
            model.half()
        if self.local_rank > -1:
            model = WrappedDDP(model)
        elif self.n_gpu > 1:
            model = WrappedDataParallel(model)

        do_stopping = False
        evalnr = 0
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
                if self.evaluator_dev is not None:
                    if self.global_step != 0 and (
                        self.global_step % self.evaluate_every == 0
                    ):
                        evalnr += 1
                        result = self.evaluator_dev.eval(model)
                        self.evaluator_dev.log_results(result, "Dev", self.global_step)
                        if self.early_stopping:
                            do_stopping, save_model, eval_value = self.early_stopping.check_stopping(result)
                            if save_model:
                                logger.info(
                                    "Saving current best model to {}, eval={}".format(
                                        self.early_stopping.save_dir, eval_value))
                                model.save(self.early_stopping.save_dir)
                                self.data_silo.processor.save(self.early_stopping.save_dir)
                            if do_stopping:
                                # log the stopping
                                logger.info("STOPPING EARLY AT EPOCH {}, STEP {}, EVALUATION {}".format(epoch, step, evalnr))
                if do_stopping:
                    break
                self.global_step += 1
            if do_stopping:
                break
        if self.evaluator_test is not None:
            if self.early_stopping and self.early_stopping.save_dir:
                logger.info("Restoring best model so far from {}".format(self.early_stopping.save_dir))
                lm_name = model.language_model.name
                model = AdaptiveModel.load(self.early_stopping.save_dir, self.device, lm_name=lm_name)
                model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)
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
