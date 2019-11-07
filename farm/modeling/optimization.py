from importlib import import_module
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

try:
    from apex import amp
    try:
        from apex.parallel import convert_syncbn_model
        APEX_PARALLEL_AVAILABLE = True
    except AttributeError:
        APEX_PARALLEL_AVAILABLE = False
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


def initialize_optimizer(model,
                         optim_opts,
                         n_batches,
                         n_epochs,
                         device,
                         distributed=False,
                         grad_acc_steps=1,
                         local_rank=-1,
                         sched_opts=None,
                         use_amp=None,
                         warmup_proportion=0.1):

    num_train_optimization_steps = calculate_optimization_steps(
        n_batches, grad_acc_steps, n_epochs, local_rank
    )

    # Log params
    MlLogger.log_params(
        {
            "learning_rate": optim_opts['lr'] if 'lr' in optim_opts else 'optimizer default',
            "warmup_proportion": warmup_proportion,
            "use_amp": use_amp,
            "num_train_optimization_steps": num_train_optimization_steps,
        }
    )

    optimizer = _get_optim(model, optim_opts)

    scheduler = None
    if sched_opts:
        if 'warmup_steps' not in sched_opts:
            sched_opts['warmup_steps'] = int(warmup_proportion * num_train_optimization_steps)
        if 't_total' not in sched_opts:
            sched_opts['t_total'] = num_train_optimization_steps
        scheduler = _get_scheduler(optimizer, sched_opts)

    model, optimizer = _optimize_model(model, device, local_rank, optimizer, distributed, use_amp)

    return model, optimizer, scheduler


def _get_optim(model, opts):
    """ Get the optimizer based on dictionary with options. Options are passed to the optimizer constructor.

    :param model: model to optimize
    :param opts: config dictionary. MUST contain the 'name' key, which will be looked up in torch.optim.lr_scheduler,
    transformers.optimization, or apex.optimizers. 'no_decay' can be given. Parameters containing any of those strings
    will have weight_decay set to 0.
    :return: created optimizer
    """
    optim_name = opts.pop('name')
    weight_decay = opts.pop('weight_decay', None)
    no_decay = opts.pop('no_decay', None)

    if no_decay:
        optimizable_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             **opts},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0,
             **opts}
        ]
    else:
        optimizable_parameters = [{'params': [p for p in model.parameters() if p.requires_grad], **opts}]

    # default weight decay is not the same for all optimizers, so we can't use default value
    # only explicitly add weight decay if it's given
    if weight_decay is not None:
        optimizable_parameters[0]['weight_decay'] = weight_decay

    try:
        optim_constructor = getattr(import_module('torch.optim'), optim_name)
    except AttributeError:
        try:
            optim_constructor = getattr(import_module('transformers.optimization'), optim_name)
        except AttributeError:
            try:
                optim_constructor = getattr(import_module('apex.optimizers'), optim_name)
            except (AttributeError, ImportError):
                raise AttributeError(f"Optimizer '{optim_name}' not found in 'torch', 'transformers', or 'apex'")

    logger.info(f"Using optimizer '{optim_name}'")
    return optim_constructor(optimizable_parameters)


def _get_scheduler(optimizer, opts):
    """ Get the scheduler based on dictionary with options. Options are passed to the scheduler constructor.

    :param optimizer: optimizer whose learning rate to control
    :param opts: config dictionary. MUST contain the 'name' key, which will be looked up in torch.optim.lr_scheduler
    or transformers.optimization
    :return: created scheduler
    """
    sched_name = opts.pop('name')

    try:
        sched_constructor = getattr(import_module('torch.optim.lr_scheduler'), sched_name)
    except AttributeError:
        try:
            sched_constructor = getattr(import_module('transformers.optimization'), sched_name)
        except AttributeError:
            raise AttributeError(f"Scheduler '{sched_name}' not found in 'torch' or 'transformers'")

    logger.info(f"Using scheduler '{sched_name}'")
    return sched_constructor(optimizer, **opts)


def calculate_optimization_steps(n_batches, grad_acc_steps, n_epochs, local_rank):
    optimization_steps = int(n_batches / grad_acc_steps) * n_epochs
    if local_rank != -1:
        optimization_steps = optimization_steps // torch.distributed.get_world_size()
    logger.info(f"Number of optimization steps: {optimization_steps:,}")
    return optimization_steps


def _optimize_model(model, device, local_rank, optimizer=None, distributed=False, use_amp=None):
    model, optimizer = _init_amp(model, device, optimizer, use_amp)

    if distributed:
        if APEX_PARALLEL_AVAILABLE:
            model = convert_syncbn_model(model)

        n = torch.cuda.device_count() // dist.get_world_size()
        device_ids = list(range(local_rank * n, (local_rank + 1) * n))
        # for some models DistributedDataParallel might complain about parameters
        # not contributing to loss. find_used_parameters remedies that.
        model = DistributedDataParallel(model,
                                        device_ids=device_ids,
                                        output_device=device_ids[0],
                                        find_unused_parameters=True)

    return model, optimizer


def _init_amp(model, device, optimizer=None, use_amp=None):
    model = model.to(device)
    if use_amp and optimizer:
        model, optimizer = amp.initialize(model, optimizer, opt_level=use_amp)

    return model, optimizer
