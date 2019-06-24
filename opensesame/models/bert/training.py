from __future__ import absolute_import, division, print_function

import logging
import os
import numpy as np
import torch
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from seqeval.metrics import classification_report as token_classification_report
from seqeval.metrics import f1_score
from mlflow import log_metric, log_param, log_artifact, set_tracking_uri, set_experiment, start_run

from opensesame.file_utils import OPENSESAME_CACHE, WEIGHTS_NAME, CONFIG_NAME, read_config
from opensesame.models.bert.modeling import BertForSequenceClassification, BertForTokenClassification
from opensesame.models.bert.tokenization import BertTokenizer
from opensesame.models.bert.optimization import BertAdam, WarmupLinearSchedule
from opensesame.data_handler.general import BertDataBunch
from opensesame.metrics import compute_metrics
from opensesame.utils import set_all_seeds, initialize_device_settings

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


class Trainer:
    def __init__(self, optimizer, data_bunch, epochs, n_gpu, grad_acc_steps, fp16, learning_rate, warmup_linear, warmup_proportion, device, evaluate_every=100):
        self.data_bunch = data_bunch
        self.epochs = int(epochs)
        self.evaluator_dev = Evaluator(data_bunch.dev_data_loader, data_bunch.)
        self.evaluator_test = Evaluator(data_bunch.test_data_loader)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.warmup_linear = warmup_linear
        self.global_step = 0
        self.data_loader_train = data_bunch.train_data_loader
        self.device = device

    def train(self, model):
        logger.info("***** Running training *****")
        model.train()
        for epoch_curr in trange(self.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(self.data_loader_train, desc="Iteration")):

                # TODO: Can this be handled by Dataloader somehow?
                # Move batch of samples to device
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # Forward pass through model
                loss = model.forward_loss(input_ids=input_ids,
                                          token_type_ids=segment_ids,
                                          attention_mask=input_mask,
                                          labels=label_ids)
                self.backward_propagate(loss, step)

                # Perform evaluation
                if self.global_step % self.evaluate_every == 1:
                    logger.info("Eval after step: {}".format(self.global_step))
                    self.evaluator_dev.eval(model)

                    # evaluation(model=model, data_bunch=self.data_bunch, output_mode=, device=device,
                    #            metric="acc", data_type="dev", global_step=self.global_step)

                self.global_step += 1
        return model

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
        # adjust loss for multi-gpu and distributed calculations
        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss


class Evaluator:
    def __init__(self, data_loader, dataset, label_list, device, metric, eval_type):

        self.data_loader = data_loader
        self.label_map = {i: label for i, label in enumerate(label_list)}
        self.loss_all = []
        self.logits_all = []
        self.preds_all = []
        self.Y_all = dataset.tensors[3]
        self.device = device
        # Where should metric be defined? When dataset loaded? In config?
        self.metric = metric

        if eval_type == "token_classification":
            self.classification_report = token_classification_report
        elif eval_type == "sequence_classification":
            self.classification_report = classification_report
        else:
            raise NotImplementedError

    def eval(self, model):
        """ This model is currently written idealistically from the high level so that it is clean
        but might not function when performing NER """
        model.eval()
        for step, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids=input_ids,
                               token_type_ids=segment_ids,
                               attention_mask=input_mask)
                loss = model.logits_to_loss(logits=logits,
                                             labels=label_ids)
                preds = model.logits_to_preds(logits)


            self.loss_all += list(loss.numpy())
            self.logits_all += list(logits.numpy())
            self.preds_all += list(preds.numpy())


        logger.info("***** Eval results *****")

        loss_total = np.sum(self.loss_all)
        loss_eval = np.mean(self.loss_all)

        result = {"loss_eval": loss_eval}
        result[self.metric] = compute_metrics(self.metric, self.preds_all, self.Y_all)
        report = self.classification_report(self.preds_all, self.Y_all, digits=4)
        logger.info("\n%s", report)


def evaluation_old(model, data_bunch, output_mode, device, metric, data_type, report_output_dir=None,
                   global_step=None, n_gpu=None):
    # TODO: need to differentiate between args.do_eval for standalone eval and eval within the training loop
    label_map = {i: label for i, label in enumerate(data_bunch.label_list)}

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    y_true_ner = []

    # TODO: This could get confusing since sometimes test == dev. No test shouldnt be dev!!
    # Can this be handled elsewhere?
    if data_type == "dev":
        logger.info("***** Running evaluation on dev *****")
        data_loader = data_bunch.dev_data_loader
        dataset = data_bunch.dev_dataset
    elif data_type == "test":
        logger.info("***** Running evaluation on test *****")
        data_loader = data_bunch.test_data_loader
        dataset = data_bunch.test_dataset

    for input_ids, input_mask, segment_ids, label_ids in tqdm(data_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           token_type_ids=segment_ids,
                           attention_mask=input_mask,
                           labels=None)
            tmp_eval_loss = model.logits_to_loss(logits=logits,
                                                 labels=label_ids,
                                                 attention_mask=input_mask)

        eval_loss += tmp_eval_loss.mean().item()

        if output_mode == "ner":
            batch_preds, batch_y_true = ignore_subword_tokens(logits, input_mask, label_map, label_ids)
            preds += batch_preds
            y_true_ner += batch_y_true
        else:
            if nb_eval_steps == 0:
                # overwrite preds of type list (instantiated above) to handle multidimensional data better
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.concatenate((preds, logits.detach().cpu().numpy()), axis=0)

        nb_eval_steps += 1
        # end of eval batch loop

    if output_mode == "ner":
        y_true = y_true_ner
    else:
        y_true = dataset.tensors[3]

    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(metric, preds, y_true)
    result['eval_loss'] = eval_loss

    # TODO report currently only works for classification + NER, does it work for regression?
    logger.info("***** Eval results *****")

    if output_mode == "ner":
        f1 = f1_score(y_true, preds, average='micro')
        logger.info("F1 Score: {}".format(f1))
        result["F1"] = f1
        report = token_classification_report(preds, y_true, digits=4)
    else:
        report = classification_report(preds, y_true.numpy(), digits=4)
    logger.info("\n%s", report)

    # Log to mlflow
    # TODO make it optional
    for metric_name, metric_val in result.items():
        log_metric("{} {}".format(data_type, metric_name), metric_val, step=global_step)

    if report_output_dir:
        output_eval_file = os.path.join(report_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(report)

def train_old(model, optimizer, data_bunch, device, output_mode, n_gpu, warmup_linear, args, metric):
#TODO get rid of "args" here

    # Begin train loop
    global_step = 0
    logger.info("***** Running training *****")
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(data_bunch.train_data_loader, desc="Iteration")):
            # get loss for single batch
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            loss = model(input_ids = input_ids,
                         token_type_ids = segment_ids,
                         attention_mask = input_mask,
                         labels = label_ids)


            # adjust loss for multi-gpu and distributed calculations
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()

            # Perform evaluation
            if (args.eval_every and (global_step % args.eval_every == 1)):
                logger.info("Eval after step: {}".format(global_step))
                evaluation(model=model, data_bunch=data_bunch, output_mode=output_mode, device=device,
                           metric=metric, data_type="dev", global_step=global_step, n_gpu=n_gpu)

            global_step += 1
    return model


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
    if local_rank != -1:
        try:
            #TODO check for dataparallel problems, as in WrappedDataParallel
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = WrappedDataParallel(model)
    return model

def ignore_subword_tokens(logits, input_mask, label_map, label_ids):
    all_label_ids = []
    preds = []
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()

    for i, mask in enumerate(input_mask):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(mask):
            if j == 0:
                continue
            if m:
                if label_map[label_ids[i][j].item()] != "X":
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
            else:
                temp_1.pop()
                temp_2.pop()
                break
        all_label_ids.append(temp_1)
        preds.append(temp_2)
    return all_label_ids, preds

def calculate_optimization_steps(n_examples, batch_size, grad_acc_steps, n_epochs, local_rank):
    optimization_steps = int(n_examples / batch_size / grad_acc_steps) * n_epochs
    if local_rank != -1:
        optimization_steps = optimization_steps // torch.distributed.get_world_size()
    return optimization_steps

def balanced_class_weights(dataset):
    all_label_ids = [x[3].item() for x in dataset]
    weights = list(compute_class_weight("balanced", np.unique(all_label_ids), all_label_ids))
    logger.info("Using weighted loss for balancing classes. Weights: {}".format(weights))
    return weights


def run_model(args, prediction_head, processor, output_mode, metric):
    # Basic init and input validation
    # TODO: Is there a better place to put this?
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.mlflow_url:
        set_tracking_uri(args.mlflow_url)
        set_experiment(args.mlflow_experiment)
        start_run(run_name=args.mlflow_run_name, nested=args.mlflow_nested)
        for key in args.__dict__:
            log_param(key, args.__dict__[key])

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
    device, n_gpu = initialize_device_settings(no_cuda=args.no_cuda, local_rank=args.local_rank, fp16=args.fp16)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    set_all_seeds(args.seed)

    # Prepare Data
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    data_bunch = BertDataBunch(args.data_dir, processor, output_mode, tokenizer, args.train_batch_size,
                               args.max_seq_length, local_rank=args.local_rank)

    # Training
    if args.do_train:

        num_train_optimization_steps = calculate_optimization_steps(data_bunch.num_train_examples,
                                                                      args.train_batch_size,
                                                                      args.gradient_accumulation_steps,
                                                                      args.num_train_epochs,
                                                                      args.local_rank)

        # TODO: This should be an attribute within data_bunch
        # Compute class weighting to balance uneven class distribution
        weights = None
        if args.balance_classes:
            weights = balanced_class_weights(data_bunch.train_dataset)

        # Init model
        model = initialize_model(bert_model=args.bert_model,
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
        # Run actual training
        # model = train(model=model, optimizer=optimizer, data_bunch=data_bunch, device=device, output_mode=output_mode,
        #               n_gpu=n_gpu,warmup_linear= warmup_linear, args=args, metric=metric)

        trainer = Trainer(optimizer=optimizer,
                          data_bunch=data_bunch,
                          epochs=args.num_train_epochs,
                          n_gpu=n_gpu,
                          grad_acc_steps=args.gradient_accumulation_steps,
                          fp16=args.fp16,
                          learning_rate=args.learning_rate,
                          warmup_linear=warmup_linear,
                          warmup_proportion=args.warmup_proportion,
                          evaluate_every=args.eval_every,
                          device=device)
        model = trainer.train(model)

    # Saving and loading the model
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save_model(model, tokenizer, args)
        output_dir = args.output_dir
    else:
        output_dir = args.bert_model
    model, tokenizer = load_model(output_dir, prediction_head, args.do_lower_case, data_bunch.num_labels)
    model.to(device)

    # Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        evaluation(model=model, data_bunch=data_bunch, output_mode=output_mode, device=device,
                   metric=metric, data_type="test", report_output_dir=args.output_dir, n_gpu=n_gpu)