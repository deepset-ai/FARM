from __future__ import absolute_import, division, print_function

import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from farm.utils import MLFlowLogger as MlLogger
from farm.eval import Evaluator
from farm.data_handler.data_silo import DataSilo
from farm.visual.ascii.images import GROWING_TREE
from farm.modeling.adaptive_model import AdaptiveModel

try:
    from apex import amp
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        lr_schedule=None,
        evaluate_every=100,
        evaluator_dev=None,
        evaluator_test=None,
        use_amp=None,
        grad_acc_steps=1,
        local_rank=-1,
        early_stopping=None,
        log_learning_rate=False
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
        :param lr_schedule: An optional scheduler object that can regulate the learning rate of the optimizer
        :param evaluate_every: Perform dev set evaluation after this many steps of training.
        :type evaluate_every: int
        :param evaluator_dev: Evaluator for dev set. Options:
                              `None` (Default) => will init a new evaluator, if there's a dev set in the DataSilo
                              `Evaluator Object` => use the manually supplied evaluator
                              `False` => Don't use any evaluator
        :type evaluator_dev: Evaluator, None or False
        :param evaluator_test: Evaluator for test set. Options:
                              `None` (Default) => will init a new evaluator, if there's a test set in the DataSilo
                              `Evaluator Object` => use the manually supplied evaluator
                              `False` => Don't use any evaluator
        :type evaluator_test: Evaluator, None or False
        :param use_amp: Whether to use automatic mixed precision with Apex. One of the optimization levels must be chosen.
                        "O1" is recommended in almost all cases.
        :type use_amp: str
        :param grad_acc_steps: TODO
        :type grad_acc_steps: int
        :param local_rank: TODO
        :type local_rank: int
        :param early_stopping: an initialized EarlyStopping object to control early stopping and saving of best models.
        :type early_stopping: EarlyStopping
        :param log_learning_rate: Whether to log learning rate to Mlflow
        :type log_learning_rate: bool
        """
        self.data_silo = data_silo
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.use_amp = use_amp
        self.lr_schedule = lr_schedule
        self.global_step = 0
        self.data_loader_train = data_silo.get_data_loader("train")
        self.device = device
        self.local_rank = local_rank
        self.log_params()
        self.early_stopping = early_stopping
        self.log_learning_rate = log_learning_rate

        if use_amp and not AMP_AVAILABLE:
            raise ImportError(f'Got use_amp = {use_amp}, but cannot find apex. '
                              'Please install Apex if you want to make use of automatic mixed precision. '
                              'https://github.com/NVIDIA/apex')

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

        do_stopping = False
        evalnr = 0
        loss = 0
        for epoch in range(1, self.epochs + 1):
            progress_bar = tqdm(self.data_loader_train)
            for step, batch in enumerate(progress_bar):
                progress_bar.set_description(f"Train epoch {epoch}/{self.epochs} (Cur. train loss: {loss:.4f})")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward pass through model
                logits = model.forward(**batch)
                per_sample_loss = model.logits_to_loss(logits=logits, **batch)

                loss = self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if self.evaluator_dev:
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

        # With early stopping we want to restore the best model
        if self.early_stopping and self.early_stopping.save_dir:
            logger.info("Restoring best model so far from {}".format(self.early_stopping.save_dir))
            lm_name = model.language_model.name
            model = AdaptiveModel.load(self.early_stopping.save_dir, self.device, lm_name=lm_name)
            model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Eval on test set
        if self.evaluator_test:
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
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.log_learning_rate:
            MlLogger.log_metrics({"learning_rate": self.lr_schedule.get_lr()[0]}, step=self.global_step)

        if (step + 1) % self.grad_acc_steps == 0:
            # TODO We might wanna add gradient clipping here
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_schedule:
                self.lr_schedule.step()
        return loss

    def adjust_loss(self, loss):
        loss = loss.mean()
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss

    def log_params(self):
        params = {"epochs": self.epochs, "n_gpu": self.n_gpu, "device": self.device}
        MlLogger.log_params(params)
