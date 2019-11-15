from importlib import import_module
import logging
import sys

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel

# Used indirectly in _get_optim() to avoid name collision with torch's AdamW
from transformers.optimization import AdamW as TransformersAdamW

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


class WrappedDataParallel(DataParallel):
    """
    A way of adapting attributes of underlying class to parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see: https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class WrappedDDP(DistributedDataParallel):
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

    # Get optimizer from pytorch, transformers or apex
    optimizer = _get_optim(model, optim_opts)

    # Get learning rate schedule
    scheduler = None
    if sched_opts:
        if 'warmup_steps' not in sched_opts:
            sched_opts['warmup_steps'] = int(warmup_proportion * num_train_optimization_steps)
        if 't_total' not in sched_opts:
            sched_opts['t_total'] = num_train_optimization_steps
        scheduler = _get_scheduler(optimizer, sched_opts)

    # Adjust for parallel training + amp
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

    # Import optimizer by checking in order: torch, transformers, apex and local imports
    try:
        optim_constructor = getattr(import_module('torch.optim'), optim_name)
    except AttributeError:
        try:
            optim_constructor = getattr(import_module('transformers.optimization'), optim_name)
        except AttributeError:
            try:
                optim_constructor = getattr(import_module('apex.optimizers'), optim_name)
            except (AttributeError, ImportError):
                try:
                    # Workaround to allow loading AdamW from transformers even though pytorch > 1.2 has now also a AdamW
                    optim_constructor = getattr(sys.modules[__name__], optim_name)
                except (AttributeError, ImportError):
                    raise AttributeError(f"Optimizer '{optim_name}' not found in 'torch', 'transformers', 'apex' or 'local imports")

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
            # TODO this will switch soon in transformers: https://github.com/huggingface/transformers/pull/1832
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

        n_gpu = torch.cuda.device_count() // torch.distributed.get_world_size()
        device_ids = list(range(local_rank * n_gpu, (local_rank + 1) * n_gpu))
        # for some models DistributedDataParallel might complain about parameters
        # not contributing to loss. find_used_parameters remedies that.
        #TODO check if Wrapped DDP still needed?
        model = DistributedDataParallel(model,
                                        device_ids=device_ids,
                                        output_device=device_ids[0],
                                        find_unused_parameters=True)

    elif torch.cuda.device_count() > 1:
        model = WrappedDataParallel(model)

    return model, optimizer


def _init_amp(model, device, optimizer=None, use_amp=None):
    model = model.to(device)
    if use_amp and optimizer:
        model, optimizer = amp.initialize(model, optimizer, opt_level=use_amp)

    return model, optimizer
