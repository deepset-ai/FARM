import logging
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import numpy
import shutil

from farm.utils import MLFlowLogger as MlLogger
from farm.utils import GracefulKiller
from farm.eval import Evaluator
from farm.data_handler.data_silo import DataSilo
from farm.visual.ascii.images import GROWING_TREE
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import get_scheduler

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
        model,
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
        log_learning_rate=False,
        checkpoint_on_sigterm=False,
        checkpoint_every=None,
        checkpoint_root_dir=None,
        checkpoints_to_keep=3,
        from_epoch=0,
        from_step=0,
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
        :param checkpoint_on_sigterm: save a checkpoint for the Trainer when a SIGTERM signal is sent. The checkpoint
               can be used to resume training. It is useful in frameworks like AWS SageMaker with Spot instances where
               a SIGTERM notifies to save the training state and subsequently the instance is terminated.
        :type checkpoint_on_sigterm: bool
        :param checkpoint_every: save a train checkpoint after this many steps of training.
        :type checkpoint_every: int
        :param checkpoint_root_dir: the Path of directory where all train checkpoints are saved. For each individual
               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
        :type checkpoint_root_dir: Path
        :param checkpoints_to_keep: maximum number of train checkpoints to save.
        :type checkpoints_to_keep: int
        :param from_epoch: the epoch number to start the training from. In the case when training resumes from a saved
               checkpoint, it is used to fast-forward training to the last epoch in the checkpoint.
        :type from_epoch: int
        :param from_step: the step number to start the training from. In the case when training resumes from a saved
               checkpoint, it is used to fast-forward training to the last step in the checkpoint.
        :type from_step: int
        """

        self.model = model
        self.data_silo = data_silo
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.use_amp = use_amp
        self.lr_schedule = lr_schedule
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
        self.checkpoint_on_sigterm = checkpoint_on_sigterm
        if checkpoint_on_sigterm:
            self.sigterm_handler = GracefulKiller()
        else:
            self.sigterm_handler = None
        self.checkpoint_root_dir = checkpoint_root_dir
        self.checkpoints_to_keep = checkpoints_to_keep
        self.checkpoint_every = checkpoint_every
        if self.checkpoint_every and not checkpoint_root_dir:
            raise Exception("checkpoint_path needs to be supplied when using checkpoint_every.")
        if checkpoint_on_sigterm and not checkpoint_root_dir:
            raise Exception("checkpoint_path needs to be supplied when using checkpoint_on_sigterm.")

        self.from_epoch = from_epoch
        self.from_step = from_step
        self.global_step = (from_epoch * from_step) - 1

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

    @classmethod
    def create_or_load_checkpoint(cls, data_silo, checkpoint_root_dir, resume_from_checkpoint="latest", **kwargs):
        """
        Try loading a saved Trainer checkpoint. If no checkpoint found, it creates a new instance of Trainer.

        :param data_silo: A DataSilo object that will contain the train, dev and test datasets as PyTorch DataLoaders
        :type data_silo: DataSilo
        :param checkpoint_root_dir: Path of the directory where all train checkpoints are saved. Each individual
               checkpoint is stored in a sub-directory under it.
        :type checkpoint_root_dir: Path
        :param resume_from_checkpoint: the checkpoint name to start training from, e.g., "epoch_1_step_4532". It
               defaults to "latest", using the checkpoint with the highest train steps.
        :type resume_from_checkpoint: str
        """
        checkpoint_to_load = None
        if checkpoint_root_dir.exists():
            if resume_from_checkpoint == "latest":
                saved_checkpoints = cls._get_checkpoints(checkpoint_root_dir)
                if saved_checkpoints:
                    checkpoint_to_load = saved_checkpoints[0][0]  # latest checkpoint
                else:
                    checkpoint_to_load = None
            else:
                checkpoint_to_load = checkpoint_root_dir / resume_from_checkpoint

        if checkpoint_to_load:
            trainer = cls._load_checkpoint(path=checkpoint_to_load, data_silo=data_silo)
            logging.info(f"Resuming training from the train checkpoint at {checkpoint_to_load} ...")
        else:
            logging.info(f"No train checkpoints found. Starting a new training ...")
            trainer = Trainer(data_silo=data_silo, checkpoint_root_dir=checkpoint_root_dir, **kwargs)
        return trainer

    @classmethod
    def _load_checkpoint(cls, path, data_silo):
        """
        Load the train checkpoint at given path.

        :param path: The checkpoint path is subdirectory under checkpoint_root_dir. The individual checkpoint dirs have
               a default naming convention of "epoch_{epoch_num}_step_{step_num}".
        :type path: Path
        :param data_silo: A DataSilo object that will contain the train, dev and test datasets as PyTorch DataLoaders
        :type data_silo: DataSilo
        """
        if not path.exists():
            raise Exception(f"The checkpoint path {path} does not exists.")

        trainer_checkpoint = torch.load(path / "trainer")
        trainer_state_dict = trainer_checkpoint["trainer_state_dict"]

        # Just setting seeds is not sufficient to have deterministic results when resuming
        # training from a checkpoint. Additionally, the previous states of Random Number
        # Generators also need to be restored from the saved checkpoint.
        numpy_rng_state = trainer_checkpoint["numpy_rng_state"]
        numpy.random.set_state(numpy_rng_state)
        rng_state = trainer_checkpoint["rng_state"]
        cuda_rng_state = trainer_checkpoint["cuda_rng_state"]
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

        model = trainer_checkpoint["model"]

        optimizer = trainer_checkpoint["optimizer"]

        scheduler_state_dict = trainer_checkpoint["scheduler_state"]
        scheduler_opts = trainer_checkpoint["scheduler_opts"]
        scheduler_opts["last_epoch"] = scheduler_state_dict["last_epoch"]
        scheduler = get_scheduler(optimizer, scheduler_opts)
        scheduler.load_state_dict(scheduler_state_dict)

        trainer = Trainer(
            data_silo=data_silo,
            model=model,
            optimizer=optimizer,
            lr_schedule=scheduler,
            **trainer_state_dict
        )

        logger.info(f"Loaded a train checkpoint from {path}")
        return trainer

    @classmethod
    def _get_checkpoints(cls, checkpoint_root_dir):
        """
        Get a list of checkpoints sorted by the number of training steps.
        """
        dirs = [d for d in checkpoint_root_dir.iterdir() if d.is_dir() and d.name.startswith("epoch")]

        checkpoints_with_total_steps = []
        for d in dirs:
            epoch, step = [int(s) for s in str(d).split("_") if s.isdigit()]
            checkpoints_with_total_steps.append((d, (epoch + 1) * (step + 1)))
        checkpoints_with_total_steps.sort(key=lambda tup: tup[1], reverse=True)

        return checkpoints_with_total_steps

    def _save(self):
        """
        Save a train checkpoint at the Trainer's checkpoint_path.

        Some objects(eg, scheduler) in the Trainer are not serializable using the Pickle module. For these objects,
        the state_dict is stored for the checkpoint, that can be used to reconstruct a similar state upon resuming
        train from the checkpoint.

        #TODO The model is currently saved as a whole serialized object. The disadvantage of this approach is that it is
        bound to specifics Python version, FARM version, directory structures etc. A more modular and reusable approach
        is to save using AdaptiveModel's save() method where the model and the state_dict are stored separately.

        # TODO custom defined evaluators are not saved in the checkpoint.
        """
        checkpoint_path = self.checkpoint_root_dir / "checkpoint_in_progress"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        trainer_state_dict = self._get_state_dict()
        self.model.save(checkpoint_path)
        torch.save(
            {
                "model": self.model,
                "trainer_state_dict": trainer_state_dict,
                "model_state_dict": self.model.state_dict(),
                "optimizer": self.optimizer,
                "scheduler_opts": self.lr_schedule.opts,
                "scheduler_state": self.lr_schedule.state_dict(),
                "numpy_rng_state": numpy.random.get_state(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
            },
            checkpoint_path / "trainer",
        )

        checkpoint_name = f"epoch_{self.from_epoch}_step_{self.from_step}"
        checkpoint_path.replace(Path(checkpoint_path.parent) / checkpoint_name)

        saved_checkpoints = self._get_checkpoints(self.checkpoint_root_dir)
        if len(saved_checkpoints) > self.checkpoints_to_keep:
            for cp, _ in saved_checkpoints[self.checkpoints_to_keep:]:
                shutil.rmtree(cp)

        logger.info(f"Saved a training checkpoint at {checkpoint_name}")

    def _get_state_dict(self):
        """
        Serializable state dictionary of a Trainer object
        """
        state_dict = {
            "evaluate_every": self.evaluate_every,
            "n_gpu": self.n_gpu,
            "grad_acc_steps": self.grad_acc_steps,
            "device": self.device,
            "local_rank": self.local_rank,
            "early_stopping": self.early_stopping,
            "epochs": self.epochs,
            "checkpoint_on_sigterm": self.checkpoint_on_sigterm,
            "checkpoint_root_dir": self.checkpoint_root_dir,
            "checkpoint_every": self.checkpoint_every,
            "from_epoch": self.from_epoch,
            "from_step": self.from_step,
            "log_learning_rate": self.log_learning_rate,
        }

        return state_dict

    def train(self):
        """ Perform the training procedure. """

        # connect the prediction heads with the right output from processor
        self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Check that the tokenizer fits the language model
        self.model.verify_vocab_size(vocab_size=len(self.data_silo.processor.tokenizer))

        logger.info(f"\n {GROWING_TREE}")
        self.model.train()

        do_stopping = False
        evalnr = 0
        loss = 0

        resume_from_step = self.from_step

        for epoch in range(self.from_epoch + 1, self.epochs + 1):
            progress_bar = tqdm(self.data_loader_train)
            for step, batch in enumerate(progress_bar, start=1):
                # when resuming training from a checkpoint, we want to fast forward to the step of the checkpoint
                if resume_from_step and step <= resume_from_step:
                    if resume_from_step == step:
                        resume_from_step = None
                    continue

                if self.sigterm_handler and self.sigterm_handler.kill_now:  # save the current state as a checkpoint
                    logger.info("Received a SIGTERM signal. Saving the current train state as a checkpoint ...")
                    self._save()
                    sys.exit(0)

                # save a checkpoint and continue train (do not create a new checkpoint if just resumed from a checkpoint)
                if self.checkpoint_every and step % self.checkpoint_every == 0 and resume_from_step + 1 != step:
                    self._save()

                progress_bar.set_description(f"Train epoch {epoch}/{self.epochs} (Cur. train loss: {loss:.4f})")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward pass through model
                logits = self.model.forward(**batch)
                per_sample_loss = self.model.logits_to_loss(logits=logits, global_step=self.global_step, **batch)

                loss = self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if self.evaluator_dev:
                    if self.global_step != 0 and (
                        self.global_step % self.evaluate_every == 0
                    ):
                        evalnr += 1
                        result = self.evaluator_dev.eval(self.model)
                        self.evaluator_dev.log_results(result, "Dev", self.global_step)
                        if self.early_stopping:
                            do_stopping, save_model, eval_value = self.early_stopping.check_stopping(result)
                            if save_model:
                                logger.info(
                                    "Saving current best model to {}, eval={}".format(
                                        self.early_stopping.save_dir, eval_value))
                                self.model.save(self.early_stopping.save_dir)
                                self.data_silo.processor.save(self.early_stopping.save_dir)
                            if do_stopping:
                                # log the stopping
                                logger.info("STOPPING EARLY AT EPOCH {}, STEP {}, EVALUATION {}".format(epoch, step, evalnr))
                if do_stopping:
                    break
                self.global_step += 1
                self.from_step = step
            self.from_epoch = epoch
            if do_stopping:
                break

        # With early stopping we want to restore the best model
        if self.early_stopping and self.early_stopping.save_dir:
            logger.info("Restoring best model so far from {}".format(self.early_stopping.save_dir))
            lm_name = self.model.language_model.name
            model = AdaptiveModel.load(self.early_stopping.save_dir, self.device, lm_name=lm_name)
            model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Eval on test set
        if self.evaluator_test:
            result = self.evaluator_test.eval(self.model)
            self.evaluator_test.log_results(result, "Test", self.global_step)
        return self.model

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

        if step % self.grad_acc_steps == 0:
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
