from __future__ import absolute_import, division, print_function

import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from farm.utils import MLFlowLogger as MlLogger
from farm.eval import Evaluator
from farm.data_handler.data_silo import DataSilo
from farm.visual.ascii.images import GROWING_TREE, BUSH_SEP

try:
    from apex import amp
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        local_rank=-1
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
        :param evaluator_dev: The dev set Evaluator object.
        :type evaluator_dev: Evaluator
        :param evaluator_test: The test set Evaluator object.
        :type evaluator_test: Evaluator
        :param use_amp: Whether to use automatic mixed precision with Apex. One of the optimization levels must be chosen.
                        "O1" is recommended in almost all cases.
        :type use_amp: str
        :param grad_acc_steps: TODO
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
        self.check_tokenizer_lm(self.data_silo.processor.tokenizer, model.language_model)

        logger.info(f"\n {GROWING_TREE}")
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
                if self.evaluator_dev is not None:
                    if self.global_step != 0 and (
                        self.global_step % self.evaluate_every == 0
                    ):
                        result = self.evaluator_dev.eval(model)
                        self.evaluator_dev.log_results(result, "Dev", self.global_step)

                self.global_step += 1

        if self.evaluator_test is not None:
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

        if (step + 1) % self.grad_acc_steps == 0:
            # TODO We might wanna add gradient clipping here
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_schedule:
                MlLogger.log_metrics({"learning_rate": self.lr_schedule.get_lr()[0]}, step=self.global_step)
                self.lr_schedule.step()

    def adjust_loss(self, loss):
        loss = loss.mean()
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss

    def check_tokenizer_lm(self, tokenizer, lm):
        tok_vocab_len = len(tokenizer)
        # TODO make this generic for other models
        model_vocab_len = lm.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
        #model_vocab_len = lm.model.embeddings.word_embeddings.num_embeddings
        if tok_vocab_len != model_vocab_len:
            f"Tokenizer vocabulary (len: {tok_vocab_len}) does not match original language model vocabulary (len: {model_vocab_len}). Resizing embedding layer of LM accordingly"
            lm.model.resize_token_embeddings(len(tokenizer))
            model_vocab_len = lm.model.embeddings.word_embeddings.num_embeddings
        assert tok_vocab_len == model_vocab_len

    def log_params(self):
        params = {"epochs": self.epochs, "n_gpu": self.n_gpu, "device": self.device}
        MlLogger.log_params(params)
