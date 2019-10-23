import logging

from farm.data_handler.data_silo import DataSilo
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import PredictionHead
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import Processor
from farm.train import Trainer
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
    args = read_config(file, flattend=False)
    experiments = unnestConfig(args, flattened=False)
    return experiments


def run_experiment(args):

    logger.info(
        "\n***********************************************"
        f"\n************* Experiment: {args.task.name} ************"
        "\n************************************************"
    )
    ml_logger = MlLogger(tracking_uri=args.logging.mlflow_url)
    ml_logger.init_experiment(
        experiment_name=args.logging.mlflow_experiment,
        run_name=args.logging.mlflow_run_name,
        nested=args.logging.mlflow_nested,
    )

    validate_args(args)
    distributed = bool(args.general.local_rank != -1)

    # Init device and distributed settings
    device, n_gpu = initialize_device_settings(
        use_cuda=args.general.cuda,
        local_rank=args.general.local_rank,
        fp16=args.general.fp16,
    )

    args.parameter.batch_size = int(
        args.parameter.batch_size // args.parameter.gradient_accumulation_steps
    )
    if n_gpu > 1:
        args.parameter.batch_size = args.parameter.batch_size * n_gpu
    set_all_seeds(args.general.seed)

    # Prepare Data
    tokenizer = Tokenizer.load(
        args.parameter.model, do_lower_case=args.parameter.lower_case
    )

    processor = Processor.load(
        tokenizer=tokenizer,
        max_seq_len=args.parameter.max_seq_len,
        data_dir=args.general.data_dir,
        **args.task.toDict(),  # args is of type DotMap and needs conversion to std python dicts
    )

    data_silo = DataSilo(
        processor=processor,
        batch_size=args.parameter.batch_size,
        distributed=distributed,
    )

    class_weights = None
    if args.parameter.balance_classes:
        task_names = list(processor.tasks.keys())
        if len(task_names) > 1:
            raise NotImplementedError(f"Balancing classes is currently not supported for multitask experiments. Got tasks:  {task_names} ")
        class_weights = data_silo.calculate_class_weights(task_name=task_names[0])

    model = get_adaptive_model(
        lm_output_type=args.parameter.lm_output_type,
        prediction_heads=args.parameter.prediction_head,
        layer_dims=args.parameter.layer_dims,
        model=args.parameter.model,
        device=device,
        class_weights=class_weights,
        embeds_dropout_prob=args.parameter.embeds_dropout_prob,
    )

    # Init optimizer

    # TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.parameter.learning_rate,
        warmup_proportion=args.parameter.warmup_proportion,
        loss_scale=args.general.loss_scale,
        fp16=args.general.fp16,
        n_batches=len(data_silo.loaders["train"]),
        grad_acc_steps=args.parameter.gradient_accumulation_steps,
        n_epochs=args.parameter.epochs,
    )

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.parameter.epochs,
        n_gpu=n_gpu,
        grad_acc_steps=args.parameter.gradient_accumulation_steps,
        fp16=args.general.fp16,
        local_rank=args.general.local_rank,
        warmup_linear=warmup_linear,
        evaluate_every=args.logging.eval_every,
        device=device,
    )

    model = trainer.train(model)

    model_name = (
        f"{model.language_model.name}-{model.language_model.language}-{args.task.name}"
    )
    processor.save(f"{args.general.output_dir}/{model_name}")
    model.save(f"{args.general.output_dir}/{model_name}")


def get_adaptive_model(
    lm_output_type,
    prediction_heads,
    layer_dims,
    model,
    device,
    embeds_dropout_prob,
    class_weights=None,
):
    parsed_lm_output_types = lm_output_type.split(",")
    language_model = LanguageModel.load(model)

    initialized_heads = []
    for head_name in prediction_heads.split(","):
        initialized_heads.append(
            PredictionHead.create(
                prediction_head_name=head_name,
                layer_dims=layer_dims,
                class_weights=class_weights,
            )
        )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=initialized_heads,
        embeds_dropout_prob=embeds_dropout_prob,
        lm_output_types=parsed_lm_output_types,
        device=device,
    )
    return model


def validate_args(args):
    if not args.task.do_train and not args.task.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.parameter.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.parameter.gradient_accumulation_steps
            )
        )


def save_model():
    raise NotImplementedError


def load_model():
    raise NotImplementedError
