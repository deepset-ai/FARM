# fmt: off

import logging
import os
import torch

from farm.utils import set_all_seeds, initialize_device_settings
# from farm.data_handler.preprocessing_pipeline import PPCONLL03, PPGNAD, PPGermEval18Coarse, PPGermEval18Fine, PPGermEval14
from farm.data_handler.processor import GNADProcessor, CONLLProcessor, GermEval14Processor, GermEval18CoarseProcessor, GermEval18FineProcessor
from farm.data_handler.data_bunch import DataBunch
from farm.modeling.prediction_head import TextClassificationHead, TokenClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.training import Trainer, Evaluator
from farm.modeling.optimization import BertAdam, WarmupLinearSchedule
from farm.modeling.tokenization import BertTokenizer
from farm.modeling.training import WrappedDataParallel

import logging
logger = logging.getLogger(__name__)

try:
    from farm.modeling.training import WrappedDDP
except ImportError:
    logger.info("Importing Data Loader for Distributed Training failed. Apex not installed?")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def run_model(args):

    validate_args(args)
    directory_setup(output_dir=args.output_dir, do_train=args.do_train)
    distributed = bool(args.local_rank != -1)

    # Init device and distributed settings
    device, n_gpu = initialize_device_settings(use_cuda=args.cuda,
                                               local_rank=args.local_rank,
                                               fp16=args.fp16)

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    if(n_gpu > 1):
        args.batch_size = args.batch_size * n_gpu
    set_all_seeds(args.seed)

    # Prepare Data
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.lower_case)

    processor = get_processor(name=args.name,
                                 tokenizer=tokenizer,
                                 max_seq_len=args.max_seq_len,
                                 data_dir=args.data_dir)

    data_bunch = DataBunch(processor=processor,
                           batch_size=args.batch_size,
                           distributed=distributed)

    class_weights = None
    if args.balance_classes:
        class_weights = data_bunch.class_weights

    model = get_adaptive_model(lm_output_type=args.lm_output_type,
                               prediction_head=args.prediction_head,
                               layer_dims=args.layer_dims,
                               model=args.model,
                               device=device,
                               class_weights=class_weights,
                               fp16=args.fp16,
                               embeds_dropout_prob=args.embeds_dropout_prob,
                               local_rank=args.local_rank,
                               n_gpu=n_gpu)

    # Init optimizer
    num_train_optimization_steps = calculate_optimization_steps(
        n_examples=data_bunch.n_samples("train"),
        batch_size=args.batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        n_epochs=args.epochs,
        local_rank=args.local_rank)

    # TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        loss_scale=args.loss_scale,
        fp16=args.fp16,
        num_train_optimization_steps=num_train_optimization_steps)

    evaluator_dev = Evaluator(
        data_loader=data_bunch.get_data_loader("dev"),
        label_list=processor.label_list,
        device=device,
        metrics=processor.metrics)

    evaluator_test = Evaluator(
        data_loader=data_bunch.get_data_loader("test"),
        label_list=processor.label_list,
        device=device,
        metrics=processor.metrics)

    trainer = Trainer(
        optimizer=optimizer,
        data_bunch=data_bunch,
        evaluator_dev=evaluator_dev,
        epochs=args.epochs,
        n_gpu=n_gpu,
        grad_acc_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        learning_rate=args.learning_rate,  # Why is this also passed to initialize optimizer?
        warmup_linear=warmup_linear,
        evaluate_every=args.eval_every,
        device=device)

    model = trainer.train(model)

    results = evaluator_test.eval(model)
    evaluator_test.print_results(results, "Test", trainer.global_step)

    #TODO: Model Saving and Loading


def get_adaptive_model(
    lm_output_type,
    prediction_head,
    layer_dims,
    model,
    device,
    embeds_dropout_prob,
    local_rank,
    n_gpu,
    fp16=False,
    class_weights=None
):

    if(prediction_head == "TokenClassificationHead"):
        #todo change NERhead to TokenClassificationHead
        prediction_head = TokenClassificationHead(layer_dims=layer_dims)
    elif(prediction_head == "TextClassificationHead"):
        #TODO change name here too
        prediction_head = TextClassificationHead(layer_dims=layer_dims,
                                                 class_weights=class_weights)
    else:
        raise NotImplementedError

    language_model = Bert.load(model)

    # TODO where are balance class weights?
    model = AdaptiveModel(language_model=language_model,
                          prediction_heads=prediction_head,
                          embeds_dropout_prob=embeds_dropout_prob,
                          lm_output_types=lm_output_type)
    if fp16:
        model.half()
    model.to(device)

    if local_rank > -1:
        model = WrappedDDP(model)
    elif n_gpu > 1:
        model = WrappedDataParallel(model)

    return model

def get_processor(name, data_dir, tokenizer, max_seq_len):
    # todo How to deal with the file paths???
    if name == "Conll2003":
        pipeline = CONLLProcessor(data_dir=data_dir,
                                     tokenizer=tokenizer,
                                     max_seq_len=max_seq_len)
    elif name == "GNAD":
        pipeline = GNADProcessor(data_dir=data_dir,
                                  tokenizer=tokenizer,
                                  max_seq_len=max_seq_len)
    elif name == "GermEval18Coarse":
        pipeline = GermEval18CoarseProcessor(data_dir=data_dir,
                                      tokenizer=tokenizer,
                                      max_seq_len=max_seq_len)
    elif name == "GermEval18Fine":
        pipeline = GermEval18FineProcessor(data_dir=data_dir,
                                      tokenizer=tokenizer,
                                      max_seq_len=max_seq_len)
    elif name == "GermEval14":
        pipeline = GermEval14Processor(data_dir=data_dir,
                                          tokenizer=tokenizer,
                                          max_seq_len=max_seq_len)
    else:
        raise NotImplementedError

    return pipeline

def directory_setup(output_dir, do_train):
    # Setup directory
    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
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
    learning_rate,
    warmup_proportion,
    loss_scale,
    fp16,
    num_train_optimization_steps,
):
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
