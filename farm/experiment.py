import logging
import torch

from farm.data_handler.data_silo import DataSilo
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import BertAdam, WarmupLinearSchedule, initialize_optimizer
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
    distributed = bool(args.local_rank != -1)

    # Init device and distributed settings
    device, n_gpu = initialize_device_settings(
        use_cuda=args.cuda, local_rank=args.local_rank, fp16=args.fp16
    )

    args.batch_size = int(args.batch_size // args.gradient_accumulation_steps)
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
        dev_split=args.dev_split,
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
        embeds_dropout_prob=args.embeds_dropout_prob,
    )

    # Init optimizer

    # TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        loss_scale=args.loss_scale,
        fp16=args.fp16,
        n_batches=len(data_silo.loaders["train"]),
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
        local_rank=args.local_rank,
        warmup_linear=warmup_linear,
        evaluate_every=args.eval_every,
        device=device,
    )

    model = trainer.train(model)

    model_name = (
        f"{model.language_model.name}-{model.language_model.language}-{args.name}"
    )
    processor.save(f"{args.output_dir}/{model_name}")
    model.save(f"{args.output_dir}/{model_name}")


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
    return model


def validate_args(args):
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )


def save_model():
    raise NotImplementedError


def load_model():
    raise NotImplementedError
