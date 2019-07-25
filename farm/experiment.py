import logging
import os
import torch

from farm.data_handler.data_silo import DataSilo
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import BertAdam, WarmupLinearSchedule
from farm.modeling.prediction_head import PredictionHead
from farm.modeling.tokenization import BertTokenizer
from farm.data_handler.processor import Processor
from farm.train import Trainer
from farm.train import WrappedDataParallel
from farm.utils import set_all_seeds, initialize_device_settings
from farm.utils import MLFlowLogger as MlLogger
from farm.file_utils import read_config, unnestConfig

logger = logging.getLogger(__name__)

try:
    from farm.train import WrappedDDP
except ImportError:
    logger.info(
        "Importing Data Loader for Distributed Training failed. Apex not installed?"
    )


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def load_experiments(file):
    args = read_config(file, flattend=True)
    experiments = unnestConfig(args, flattened=True)
    return experiments


def run_experiment(args):
    validate_args(args)
    directory_setup(output_dir=args.output_dir, do_train=args.do_train)
    distributed = bool(args.local_rank != -1)

    # Init device and distributed settings
    device, n_gpu = initialize_device_settings(
        use_cuda=args.cuda, local_rank=args.local_rank, fp16=args.fp16
    )

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    if n_gpu > 1:
        args.batch_size = args.batch_size * n_gpu
    set_all_seeds(args.seed)

    # Prepare Data
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.lower_case)
    processor = Processor.load(
        processor_name=args.processor_name,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        data_dir=args.data_dir,
    )

    data_silo = DataSilo(
        processor=processor, batch_size=args.batch_size, distributed=distributed
    )

    class_weights = None
    if args.balance_classes:
        class_weights = data_silo.class_weights

    model = get_adaptive_model(
        lm_output_type=args.lm_output_type,
        prediction_heads=args.prediction_head,
        layer_dims=args.layer_dims,
        model=args.model,
        device=device,
        class_weights=class_weights,
        fp16=args.fp16,
        embeds_dropout_prob=args.embeds_dropout_prob,
        local_rank=args.local_rank,
        n_gpu=n_gpu,
    )

    # Init optimizer

    # TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        loss_scale=args.loss_scale,
        fp16=args.fp16,
        n_examples=data_silo.n_samples("train"),
        batch_size=args.batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        n_epochs=args.epochs,
    )

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.epochs,
        n_gpu=n_gpu,
        grad_acc_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        warmup_linear=warmup_linear,
        evaluate_every=args.eval_every,
        device=device,
    )

    model = trainer.train(model)

    model_name = (
        f"{model.language_model.name}-{model.language_model.language}-{args.name}"
    )
    processor.save(f"saved_models/{model_name}")
    model.save(f"saved_models/{model_name}")


def get_adaptive_model(
    lm_output_type,
    prediction_heads,
    layer_dims,
    model,
    device,
    embeds_dropout_prob,
    local_rank,
    n_gpu,
    fp16=False,
    class_weights=None,
):
    parsed_lm_output_types = lm_output_type.split(",")

    initialized_heads = []
    for head_name in prediction_heads.split(","):
        initialized_heads.append(
            PredictionHead.create(
                prediction_head_name=head_name,
                layer_dims=layer_dims,
                class_weights=class_weights,
            )
        )

    language_model = LanguageModel.load(model)

    # TODO where are balance class weights?
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=initialized_heads,
        embeds_dropout_prob=embeds_dropout_prob,
        lm_output_types=parsed_lm_output_types,
        device=device,
    )
    if fp16:
        model.half()

    if local_rank > -1:
        model = WrappedDDP(model)
    elif n_gpu > 1:
        model = WrappedDataParallel(model)

    return model


def directory_setup(output_dir, do_train):
    # Setup directory
    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(output_dir)
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def validate_args(args):
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )


def initialize_optimizer(
    model,
    n_examples,
    batch_size,
    n_epochs,
    warmup_proportion=0.1,
    learning_rate=2e-5,
    fp16=False,
    loss_scale=0,
    grad_acc_steps=1,
    local_rank=-1,
):
    num_train_optimization_steps = calculate_optimization_steps(
        n_examples, batch_size, grad_acc_steps, n_epochs, local_rank
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
        )
        return optimizer, None


def calculate_optimization_steps(
    n_examples, batch_size, grad_acc_steps, n_epochs, local_rank
):
    optimization_steps = int(n_examples / batch_size / grad_acc_steps) * n_epochs
    if local_rank != -1:
        optimization_steps = optimization_steps // torch.distributed.get_world_size()
    return optimization_steps


def save_model():
    raise NotImplementedError


def load_model():
    raise NotImplementedError
