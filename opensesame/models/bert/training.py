from __future__ import absolute_import, division, print_function

import logging
import os
from collections import Counter

import warnings
import numpy as np
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as token_classification_report
from seqeval.metrics import f1_score
from mlflow import log_metric, log_param, log_artifact, set_tracking_uri, set_experiment, start_run

from opensesame.file_utils import OPENSESAME_CACHE, WEIGHTS_NAME, CONFIG_NAME, read_config
from opensesame.models.bert.modeling import BertForSequenceClassification, BertForTokenClassification
from opensesame.models.bert.tokenization import BertTokenizer
from opensesame.models.bert.optimization import BertAdam, WarmupLinearSchedule
from opensesame.data_handler.general import get_data_loader
from opensesame.metrics import compute_metrics


logger = logging.getLogger(__name__)


def train(model, optimizer, train_examples, dev_examples, label_list, device,
          tokenizer, output_mode, n_gpu, num_labels, warmup_linear, args, metric):

    # Begin train loop
    global_step = 0
    logger.info("***** Loading train data ******")
    if args.local_rank == -1:
        train_sampler = RandomSampler
    else:
        train_sampler = DistributedSampler

    train_data_loader, train_dataset = get_data_loader(train_examples, label_list,
                                                       train_sampler, args.train_batch_size,
                                                       args.max_seq_length, tokenizer, output_mode)
    logger.info("***** Running training *****")
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_data_loader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # TODO define a new function to compute loss values for all output_modes

            if output_mode == "classification":
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                if "balance_classes" in args.__dict__ and args.balance_classes:
                    # TODO: Validate that fix now also balances correctly for multiclass
                    all_label_ids = [x[3].item() for x in train_dataset]
                    class_counts = Counter(all_label_ids)
                    ratios = [1 - (c / len(all_label_ids)) for c in class_counts.values()]
                    w = torch.tensor([c / (sum(ratios)) for c in ratios])
                    logger.info("Using weighted loss for balancing classes. Weights: {}".format(w))
                    loss_fct = CrossEntropyLoss(weight=w.to(device))
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            elif output_mode == "ner":
                loss = model(input_ids, segment_ids, input_mask, labels=label_ids)
            else:
                raise NotImplementedError

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
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
                #TODO we might wanna move this increment to after the evaluation
                global_step += 1

            # Perform evaluation
            if (args.eval_every and (global_step % args.eval_every == 1)):
                logger.info("Eval after step: {}".format(global_step))
                evaluation(model, dev_examples, label_list, tokenizer, output_mode, device, num_labels,
                           metric, args.eval_batch_size, args.max_seq_length, "dev", global_step=global_step)
    return model


def evaluation(model, eval_examples, label_list, tokenizer, output_mode, device, num_labels, metric,
               eval_batch_size, max_seq_length, data_type, report_output_dir=None, global_step=None):
    # TODO: need to differentiate between args.do_eval for standalone eval and eval within the training loop
    logger.info("***** Loading eval data ******")

    eval_sampler = SequentialSampler
    eval_data_loader, eval_dataset = get_data_loader(eval_examples, label_list, eval_sampler, eval_batch_size,
                                                     max_seq_length, tokenizer, output_mode)
    label_map = {i: label for i, label in enumerate(label_list)}

    logger.info("***** Running evaluation *****")

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    y_true_ner = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        elif output_mode == "ner":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        # TODO: Loss for ner is calculated in the same way as during train i.e. predictions on non-initial subtokens are included
        eval_loss += tmp_eval_loss.mean().item()


        if output_mode == "ner":
            batch_preds, batch_y_true = ignore_subword_tokens(logits, input_mask, label_map, label_ids)
            preds += batch_preds
            y_true_ner += batch_y_true
        else:
            if nb_eval_steps == 0:
                #overwrite preds of type list to handle multidimensional data better
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.concatenate((preds, logits.detach().cpu().numpy()), axis=0)

        nb_eval_steps += 1
        # end of eval batch loop

    if output_mode == "ner":
        y_true = y_true_ner
    else:
        y_true = eval_dataset.tensors[3]

    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(metric, preds, y_true)
    result['eval_loss'] = eval_loss


    # TODO report currently only works for classification
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
    #TODO make it optional
    for metric_name, metric_val in result.items():
        log_metric("{} {}".format(data_type, metric_name), metric_val, step=global_step)

    if report_output_dir:
        output_eval_file = os.path.join(report_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(report)


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


def initialize_model(bert_model, prediction_head, num_labels, device, n_gpu, cache_dir, local_rank, fp16):
    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(str(OPENSESAME_CACHE), 'distributed_{}'.format(local_rank))
    if prediction_head == "seq_classification":
        model = BertForSequenceClassification.from_pretrained(bert_model,
                  cache_dir=cache_dir,
                  num_labels=num_labels)
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
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
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

def run_model(args, prediction_head, processor, output_mode, metric):
    # Basic init and input validation
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    if args.mlflow_url:
        set_tracking_uri(args.mlflow_url)
        set_experiment(args.mlflow_experiment)
        start_run(run_name=args.mlflow_run_name)
        for key in args.__dict__:
            log_param(key, args.__dict__[key])

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Init device and distributed settings
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load examples
    train_examples = processor.get_train_examples(args.data_dir)
    dev_evamples = processor.get_dev_examples(args.data_dir)
    # TODO: How about the case when we want to run evaluation on test at the end but want to run
    #  evaluation on dev during training?
    try:
        test_examples = processor.get_test_examples(args.data_dir)
    except:
        logger.warning("Test set not found, evaluation during training and afterwards will both be performed on dev set.")
        test_examples = dev_evamples
    if args.mlflow_url:
        log_param("num_train_examples", len(train_examples))
        log_param("num_dev_examples", len(dev_evamples))
        log_param("num_test_examples", len(test_examples))

    if args.do_train:

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        model = initialize_model(bert_model=args.bert_model, prediction_head=prediction_head, device=device,
                                 num_labels=num_labels, n_gpu=n_gpu, cache_dir=args.cache_dir,
                                 local_rank=args.local_rank, fp16=args.fp16)

        optimizer, warmup_linear = initialize_optimizer(model, args.learning_rate, args.warmup_proportion,
                                                        args.loss_scale, args.fp16, num_train_optimization_steps)

        model = train(model, optimizer, train_examples, dev_evamples, label_list, device,
                      tokenizer, output_mode, n_gpu, num_labels, warmup_linear, args, metric)

    # Saving or loading the model
    # TODO check why the previous version of this code could eval on the test set, and not overwrite the model with a blank one
    # TODO look at commit hash ef47440e5eb901ebd8e0b5ad7a5911811ce352e0
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # TODO save model with proper naming
        save_model(model, tokenizer, args)
    else:
        # TODO load model with trained prediction head
        model, tokenizer_unused = load_model(args.bert_model, prediction_head, args.do_lower_case, num_labels)
        model.to(device)

    # Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        evaluation(model, test_examples, label_list, tokenizer, output_mode, device, num_labels, metric,
                   args.eval_batch_size, args.max_seq_length, "test", args.output_dir)
