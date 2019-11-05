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
import sys

import torch

from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {})

from transformers.optimization import (
    ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
    WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule, AdamW)


def initialize_optimizer(
    model,
    n_batches,
    n_epochs,
    warmup_proportion=0.1,
    learning_rate=2e-5,
    use_amp=None,
    loss_scale=None,
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
            "use_amp": use_amp,
            "num_train_optimization_steps": num_train_optimization_steps,
        }
    )
    # Prepare params for optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # Choose optimizer depending on AMP
    if use_amp is not None:
        if log_learning_rate:
            logger.warning("Logging of learning rate is currently not supported for amp!")
        try:
            from apex import amp
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use AMP."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            bias_correction=False
        )
        model, optimizer = amp.initialize(model, optimizer, opt_level=use_amp, loss_scale=loss_scale)
    else:
        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=learning_rate
        )

    # Setup LR Schedule
    # TODO: We could easily support multiple schedules here
    lr_schedule = WarmupLinearSchedule(optimizer=optimizer,
                                       warmup_steps=int(warmup_proportion*num_train_optimization_steps),
                                       t_total=num_train_optimization_steps
                                       )

    return model, optimizer, lr_schedule


def calculate_optimization_steps(
    n_batches, grad_acc_steps, n_epochs, local_rank
):
    optimization_steps = int(n_batches / grad_acc_steps) * n_epochs
    if local_rank != -1:
        optimization_steps = optimization_steps // torch.distributed.get_world_size()
    logger.info(f"Number of optimization steps: {optimization_steps}")
    return optimization_steps
