from __future__ import absolute_import, division, print_function

import logging
import os
import numpy as np
import torch
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as token_classification_report
from seqeval.metrics import f1_score

from opensesame.file_utils import OPENSESAME_CACHE, WEIGHTS_NAME, CONFIG_NAME, read_config
from opensesame.modeling.language_model import BertForSequenceClassification, BertForTokenClassification
from opensesame.modeling.tokenization import BertTokenizer
from opensesame.modeling.optimization import BertAdam, WarmupLinearSchedule
from opensesame.data_handler.input_features import examples_to_features_ner, examples_to_features_sequence
from opensesame.data_handler.data_bunch import DataBunch


from opensesame.metrics import compute_metrics
from opensesame.utils import set_all_seeds, initialize_device_settings, MLFlowLogger

logger = logging.getLogger(__name__)


class WrappedDataParallel(torch.nn.DataParallel):
    """
    Hack to get attributes of underlying class in parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see: https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

try:
    from apex.parallel import DistributedDataParallel as DDP
    class WrappedDDP(DDP):
        """
        Hack to get attributes of underlying class in distributed mode. Same as in WrappedDataParallel above.
        Even when using distributed on a single machine with multiple GPUs, apex can speed up training significantly.
        Distributed code must be launched with "python -m torch.distributed.launch --nproc_per_node=1 run_script.py"
        """
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
except ImportError:
    logger.warn("Apex not installed. If you use distributed training with local rank != -1 apex must be installed.")


class Trainer:
    def __init__(self,
                 optimizer,
                 data_bunch,
                 evaluator_dev,
                 evaluator_test,
                 epochs,
                 n_gpu,
                 grad_acc_steps,
                 fp16,
                 learning_rate,
                 warmup_linear,
                 device,
                 evaluate_every=100):
        self.data_bunch = data_bunch
        self.evaluator_dev = evaluator_dev
        self.evaluator_test = evaluator_test
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.warmup_linear = warmup_linear
        self.global_step = 0
        self.data_loader_train = data_bunch.get_data_loader("train")
        self.device = device

    def train(self, model):
        logger.info("***** Running training *****")
        model.train()
        for _ in trange(self.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(self.data_loader_train, desc="Iteration")):

                # Move batch of samples to device
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, initial_mask = batch

                # Forward pass through model
                logits = model.forward(input_ids=input_ids,
                                       token_type_ids=segment_ids,
                                       attention_mask=input_mask)
                loss = model.logits_to_loss(logits=logits,
                                            labels=label_ids,
                                            initial_mask=initial_mask,
                                            attention_mask=input_mask)
                self.backward_propagate(loss, step)

                # Perform evaluation
                if self.global_step % self.evaluate_every == 1:
                    result = self.evaluator_dev.eval(model)
                    self.print_dev(result, self.global_step)
                    # # Log to mlflow
                    # #TODO make it optional
                    # metrics = {f"dev {metric_name}": metric_val for metric_name, metric_val in result.items()}
                    # MLFlowLogger.write_metrics(metrics, step=self.global_step)

                self.global_step += 1
        return model

    def evaluate_on_test(self, model):
        result = self.evaluator_test.eval(model)
        logger.info("***** Test Eval Results *****")
        logger.info(result["report"])


    def backward_propagate(self, loss, step):
        loss = self.adjust_loss(loss)
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % self.grad_acc_steps == 0:
            if self.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = self.learning_rate * self.warmup_linear.get_lr(self.global_step, self.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()


    def adjust_loss(self, loss):
        loss = loss.mean()
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss


    @staticmethod
    def print_dev(result, step):
        logger.info("***** Dev Eval Results After Steps: {} *****".format(step))
        logger.info(result["report"])


class Evaluator:
    def __init__(self, data_loader, label_list, device, metric, output_mode, token_level):

        self.data_loader = data_loader
        self.label_map = {i: label for i, label in enumerate(label_list)}

        # These will contain the per sample loss, logits, preds and Y
        self.loss_all = []
        self.logits_all = []
        self.preds_all = []
        self.Y_all = []

        self.device = device
        # Where should metric be defined? When dataset loaded? In config?
        self.metric = metric

        # Turn classification_report into an argument of init
        if output_mode in ["classification", "ner"]:
            if token_level:
                self.classification_report = token_classification_report
            else:
                self.classification_report = classification_report
        else:
            raise NotImplementedError


    def eval(self, model):
        model.eval()

        for step, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, initial_mask = batch

            with torch.no_grad():
                logits = model.forward(input_ids=input_ids,
                                       token_type_ids=segment_ids,
                                       attention_mask=input_mask)
                loss = model.logits_to_loss(logits=logits,
                                            labels=label_ids,
                                            attention_mask=input_mask,
                                            initial_mask = initial_mask)
                # Todo BC: I don't like that Y is returned here but this is the best I have right now
                Y, preds = model.logits_to_preds(logits=logits,
                                                 input_mask=input_mask,
                                                 label_map=self.label_map,
                                                 label_ids=label_ids,
                                                 initial_mask=initial_mask)
            if Y is not None:
                label_ids = Y

            self.loss_all += list(to_numpy(loss))
            self.logits_all += list(to_numpy(logits))
            self.preds_all += list(to_numpy(preds))
            self.Y_all += list(to_numpy(label_ids))


        loss_eval = np.mean(self.loss_all)
        result = {"loss_eval": loss_eval}
        result[self.metric] = compute_metrics(self.metric, self.preds_all, self.Y_all)
        result["report"] = self.classification_report(self.Y_all, self.preds_all, digits=4)

        self.reset_state()
        return result

    def reset_state(self):
        self.loss_all = []
        self.logits_all = []
        self.preds_all = []
        self.Y_all = []

def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container

def initialize_optimizer(model, learning_rate, warmup_proportion, loss_scale, fp16, num_train_optimization_steps):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                             t_total=num_train_optimization_steps)
        return optimizer, warmup_linear

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
        return optimizer, None


def save_model(model, tokenizer, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def load_model(model_dir, prediction_head, do_lower_case, num_labels):
    # Load a trained model and vocabulary that you have fine-tuned
    if prediction_head == "seq_classification":
        model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    elif prediction_head == "simple_ner":
        model = BertForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
    else:
        raise NotImplementedError
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)
    return model, tokenizer


def initialize_model(bert_model, prediction_head, num_labels, device, n_gpu, cache_dir, local_rank, fp16, balanced_weights=None):
    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(str(OPENSESAME_CACHE), 'distributed_{}'.format(local_rank))
    if prediction_head == "seq_classification":
        model = BertForSequenceClassification.from_pretrained(bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels,
                                                              balanced_weights=balanced_weights
                                                              )
    elif prediction_head == "simple_ner":
        model = BertForTokenClassification.from_pretrained(bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    elif prediction_head == "crf_ner":
        #TODO
        raise NotImplementedError
    else:
        raise NotImplementedError

    if fp16:
        model.half()
    model.to(device)
    if local_rank > -1:
        model = WrappedDDP(model)
    elif n_gpu > 1:
        model = WrappedDataParallel(model)
    return model

def calculate_optimization_steps(n_examples, batch_size, grad_acc_steps, n_epochs, local_rank):
    optimization_steps = int(n_examples / batch_size / grad_acc_steps) * n_epochs
    if local_rank != -1:
        optimization_steps = optimization_steps // torch.distributed.get_world_size()
    return optimization_steps


# TODO: I think this might deserve its own module - it taps into so many parts of the code
def run_model(args, prediction_head, processor, output_mode, metric, token_level):
    # Basic init and input validation
    # TODO: Is there a better place to put this?
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # if args.mlflow_url:
    #     MLFlowLogger(experiment_name=args.mlflow_experiment, uri=args.mlflow_url)
    #     MLFlowLogger.init_trail(trail_name=args.mlflow_run_name, nested=args.mlflow_nested)
    #     params = {key: args.__dict__[key] for key in args.__dict__}
    #     MLFlowLogger.write_params(params)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    # Init device and distributed settings
    device, n_gpu = initialize_device_settings(use_cuda=args.cuda, local_rank=args.local_rank, fp16=args.fp16)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    set_all_seeds(args.seed)

    # Prepare Data
    tokenizer = BertTokenizer.from_pretrained(args.model,
                                              do_lower_case=args.lower_case)

    if token_level:
        ex2feat = examples_to_features_ner
    else:
        ex2feat = examples_to_features_sequence

    data_bunch = DataBunch.load(args.data_dir,
                                   processor,
                                   tokenizer,
                                   args.train_batch_size,
                                   args.max_seq_length,
                                   ex2feat,
                                   local_rank=args.local_rank)

    weights = data_bunch.get_class_weights("train")


    # Training
    if args.do_train:
        # Maybe should be in initialize optimizer though that would be very many arguments
        num_train_optimization_steps = calculate_optimization_steps(data_bunch.n_samples("train"),
                                                                      args.train_batch_size,
                                                                      args.gradient_accumulation_steps,
                                                                      args.num_train_epochs,
                                                                      args.local_rank)


        # Init model
        model = initialize_model(bert_model=args.model,
                                 prediction_head=prediction_head,
                                 device=device,
                                 num_labels=data_bunch.num_labels,
                                 n_gpu=n_gpu,
                                 cache_dir=args.cache_dir,
                                 local_rank=args.local_rank,
                                 fp16=args.fp16,
                                 balanced_weights=weights)

        # Init optimizer
        # TODO: warmup linear is sometimes NONE depending on fp16 - is there a neater way to handle this?
        optimizer, warmup_linear = initialize_optimizer(model=model,
                                                        learning_rate=args.learning_rate,
                                                        warmup_proportion=args.warmup_proportion,
                                                        loss_scale=args.loss_scale,
                                                        fp16=args.fp16,
                                                        num_train_optimization_steps=num_train_optimization_steps)

        evaluator_dev = Evaluator(data_loader=data_bunch.get_data_loader("dev"),
                                  label_list=data_bunch.label_list,
                                  device=device,
                                  metric=metric,
                                  output_mode=output_mode,
                                  token_level=token_level)

        evaluator_test = Evaluator(data_loader=data_bunch.get_data_loader("test"),
                                   label_list=data_bunch.label_list,
                                   device=device,
                                   metric=metric,
                                   output_mode=output_mode,
                                   token_level=token_level)

        trainer = Trainer(optimizer=optimizer,
                          data_bunch=data_bunch,
                          evaluator_dev=evaluator_dev,
                          evaluator_test=evaluator_test,
                          epochs=args.num_train_epochs,
                          n_gpu=n_gpu,
                          grad_acc_steps=args.gradient_accumulation_steps,
                          fp16=args.fp16,
                          learning_rate=args.learning_rate,
                          warmup_linear=warmup_linear,
                          evaluate_every=args.eval_every,
                          device=device)
        model = trainer.train(model)

    # Saving and loading the model
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save_model(model, tokenizer, args)
        output_dir = args.output_dir
    else:
        output_dir = args.model
    model, tokenizer = load_model(output_dir, prediction_head, args.lower_case, data_bunch.num_labels)
    model.to(device)

    # TODO: Model Saving and Loading
    # # Saving and loading the model
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     save_model(model, tokenizer, args)
    #     output_dir = args.output_dir
    # else:
    #     output_dir = args.bert_model
    # model, tokenizer = load_model(output_dir, prediction_head, args.do_lower_case, data_bunch.num_labels)
    # model.to(device)

    # Test set Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        trainer.evaluate_on_test(model)
