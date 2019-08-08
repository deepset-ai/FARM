# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import abc
import logging
import math
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.optimizer import required

from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {})


class _LRSchedule(ABC):
    """ Parent of all LRSchedules here. """

    warn_t_total = (
        False
    )  # is set to True for schedules where progressing beyond t_total steps doesn't make sense

    def __init__(self, warmup=0.002, t_total=-1, **kw):
        """
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps (updates) are planned
        :param kw:
        """
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            logger.warning(
                "t_total value of {} results in schedule not being applied".format(
                    t_total
                )
            )
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup)
            )
        warmup = max(warmup, 0.0)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.t_total < 0:
            return 1.0
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if (
            not nowarn
            and self.warn_t_total
            and progress > 1.0
            and progress > self.warned_for_t_total_at_progress
        ):
            logger.warning(
                "Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly.".format(
                    ret, self.__class__.__name__
                )
            )
            self.warned_for_t_total_at_progress = progress
        # end warning
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return 1.0


class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.0


class WarmupCosineSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Decreases learning rate from 1. to 0. over remaining `1 - warmup` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    warn_t_total = True

    def __init__(self, warmup=0.002, t_total=-1, cycles=0.5, **kw):
        """
        :param warmup:      see LRSchedule
        :param t_total:     see LRSchedule
        :param cycles:      number of cycles. Default: 0.5, corresponding to cosine decay from 1. at progress==warmup and 0 at progress==1.
        :param kw:
        """
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (
                1 - self.warmup
            )  # progress after warmup
            return 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    """

    def __init__(self, warmup=0.002, t_total=-1, cycles=1.0, **kw):
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(
            warmup=warmup, t_total=t_total, cycles=cycles, **kw
        )
        assert cycles >= 1.0

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (
                1 - self.warmup
            )  # progress after warmup
            ret = 0.5 * (1.0 + math.cos(math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    """
    All training progress is divided in `cycles` (default=1.) parts of equal length.
    Every part follows a schedule with the first `warmup` fraction of the training steps linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    """

    def __init__(self, warmup=0.002, t_total=-1, cycles=1.0, **kw):
        assert warmup * cycles < 1.0
        warmup = warmup * cycles if warmup >= 0 else warmup
        super(WarmupCosineWithWarmupRestartsSchedule, self).__init__(
            warmup=warmup, t_total=t_total, cycles=cycles, **kw
        )

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.0
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (
                1 - self.warmup
            )  # progress after warmup
            ret = 0.5 * (1.0 + math.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.
    """

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.0


class WarmupLinearSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """

    warn_t_total = True

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.0) / (self.warmup - 1.0), 0.0)


SCHEDULES = {
    None: ConstantLR,
    "none": ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule,
}


class BertAdam(Optimizer):
    """
    Implements BERT version of Adam algorithm with weight decay fix.
    """

    def __init__(
        self,
        params,
        lr=required,
        warmup=-1,
        t_total=-1,
        schedule="warmup_linear",
        b1=0.9,
        b2=0.999,
        e=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        log_learning_rate=False,
        **kwargs
    ):
        """
        :param params:
        :param lr: learning rate
        :param warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        :param t_total: total number of training steps for the learning rate schedule, -1  means constant learning rate
                        of 1. (no warmup regardless of warmup setting). Default: -1
        :param schedule: schedule to use for the warmup (see above). Can be `'warmup_linear'`, `'warmup_constant'`,
                         `'warmup_cosine'`, `'none'`, `None` or a `_LRSchedule` object (see below). If `None` or
                         `'none'`, learning rate is always kept constant. Default : `'warmup_linear'`
        :param b1: Adams b1. Default: 0.9
        :param b2: Adams b2. Default: 0.999
        :param e: Adams epsilon. Default: 1e-6
        :param weight_decay: Weight decay. Default: 0.01
        :param max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1)
            )
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2)
            )
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning(
                    "warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided as schedule. "
                    "Please specify custom warmup and t_total in _LRSchedule object."
                )
        defaults = dict(
            lr=lr,
            schedule=schedule,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(BertAdam, self).__init__(params, defaults)
        self.log_learning_rate = log_learning_rate

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                lr_scheduled = group["lr"]
                lr_scheduled *= group["schedule"].get_lr(state["step"])
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["next_m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["next_v"] = torch.zeros_like(p.data)

                state["step"] += 1
                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group["e"])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                lr_scheduled = group["lr"]
                lr_scheduled *= group["schedule"].get_lr(state["step"])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
        # Custom logging functionality
        if self.log_learning_rate:
            MlLogger.log_metrics({"learning_rate": lr_scheduled}, step=state["step"])
            logger.info(f'step:{state["step"]}, lr:{lr_scheduled}')
        return loss


def initialize_optimizer(
    model,
    n_batches,
    n_epochs,
    warmup_proportion=0.1,
    learning_rate=2e-5,
    fp16=False,
    loss_scale=0,
    grad_acc_steps=1,
    local_rank=-1,
    log_learning_rate=False
):
    num_train_optimization_steps = calculate_optimization_steps(
        n_batches, grad_acc_steps, n_epochs, local_rank
    )

    # Log params
    MlLogger.log_params(
        {
            "learning_rate": learning_rate,
            "warmup_proportion": warmup_proportion,
            "fp16": fp16,
            "num_train_optimization_steps": num_train_optimization_steps,
        }
    )
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if fp16:
        if log_learning_rate:
            logger.warning("Logging of learning rate is currently not supported for fp16!")
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        warmup_linear = WarmupLinearSchedule(
            warmup=warmup_proportion, t_total=num_train_optimization_steps
        )
        return optimizer, warmup_linear

    else:
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            warmup=warmup_proportion,
            t_total=num_train_optimization_steps,
            log_learning_rate=log_learning_rate
        )
        return optimizer, None


def calculate_optimization_steps(
    n_batches, grad_acc_steps, n_epochs, local_rank
):
    optimization_steps = int(n_batches / grad_acc_steps) * n_epochs
    if local_rank != -1:
        optimization_steps = optimization_steps // torch.distributed.get_world_size()
    logger.info(f"Number of optimization steps: {optimization_steps}")
    return optimization_steps