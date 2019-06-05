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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report

from opensesame.file_utils import OPENSESAME_CACHE, WEIGHTS_NAME, CONFIG_NAME, read_config
from opensesame.models.bert.modeling import BertForSequenceClassification
from opensesame.models.bert.tokenization import BertTokenizer
from opensesame.models.bert.optimization import BertAdam, WarmupLinearSchedule
from opensesame.data_handler.seq_classification import featurize_samples, get_data_loader

logger = logging.getLogger(__name__)


#TODO move these functions to a metrics.py in the package
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(metric, preds, labels):
    assert len(preds) == len(labels)
    if metric == "mcc":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif metric == "acc":
        return {"acc": simple_accuracy(preds, labels)}
    elif metric == "acc_f1":
        return acc_and_f1(preds, labels)
    elif metric == "pear_spear":
        return pearson_and_spearman(preds, labels)
    else:
        raise KeyError(metric)

def train(model, optimizer, train_examples, dev_examples, label_list, device,
          tokenizer, output_mode, num_train_optimization_steps,
          n_gpu, num_labels, warmup_linear, args, metric):

    # Begin train loop
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # TODO: Should this if statement be here?
    if args.do_train:

        logger.info("***** Loading train data ******")

        if args.local_rank == -1:
            train_sampler = RandomSampler
        else:
            train_sampler = DistributedSampler

        dev_sampler = SequentialSampler

        train_data_loader, train_dataset = get_data_loader(train_examples, label_list,
                                                           train_sampler, args.train_batch_size,
                                                           args, tokenizer, output_mode)

        logger.info("***** Running training *****")

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_data_loader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    if "balance_classes" in args.__dict__ and args.balance_classes:
                        # TODO: Validate that fix now also balances correctly for multiclass
                        all_label_ids = [x[3].item() for x in train_dataset]
                        class_counts = Counter(all_label_ids)
                        ratios = [1 - (c / len(all_label_ids)) for c in class_counts.values()]
                        w = torch.tensor([c / (sum(ratios)) for c in ratios])
                        logger.info("Using weighted loss with weigts: {}".format(w))
                        loss_fct = CrossEntropyLoss(weight=w.to(device))
                    else:
                        loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

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
                    global_step += 1

                # Perform evaluation
                if (args.eval_every and (global_step % args.eval_every == 1)):
                    evaluation(model, dev_examples, label_list, tokenizer, output_mode, device, num_labels, metric, args)


    return model

def evaluation(model, eval_examples, label_list, tokenizer, output_mode, device, num_labels, metric, args):
    # TODO: If this is false, this function does nothing
    # TODO: need to differentiate between args.do_eval for standalone eval and eval within the trianing loop
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        logger.info("***** Loading eval data ******")

        eval_sampler = SequentialSampler
        eval_data_loader, eval_dataset = get_data_loader(eval_examples, label_list, eval_sampler, args.eval_batch_size, args, tokenizer, output_mode)

        logger.info("***** Running evaluation *****")

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

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

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        all_label_ids = eval_dataset.tensors[3]
        result = compute_metrics(metric, preds, all_label_ids.numpy())

        result['eval_loss'] = eval_loss

        report = classification_report(preds, all_label_ids.numpy(), digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)


def initialize_optimizer(model, args, num_train_optimization_steps):
    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
            return optimizer, warmup_linear

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
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

def load_model(model_dir, do_lower_case, num_labels):
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)
    return model, tokenizer

def initialize_model(device, num_labels, n_gpu, args):
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(OPENSESAME_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model

def run_seq_classification(args, processor, output_mode, metric):
    # Basic init and input validation
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
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
    # TODO: How about the case when we want to run evaluation on test at the end but want to run
    #  evaluation on dev during training?
    if os.path.isfile(args.data_dir + "/test.tsv"):
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        warnings.warn("Test set not found, evaluation performed on dev set.")
        eval_examples = processor.get_dev_examples(args.data_dir)

    if args.do_train:

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        model = initialize_model(device, num_labels, n_gpu, args)

        optimizer, warmup_linear = initialize_optimizer(model, args, num_train_optimization_steps)

        model = train(model, optimizer, train_examples, eval_examples, label_list, device,
                      tokenizer, output_mode, num_train_optimization_steps,
                      n_gpu, num_labels, warmup_linear, args, metric)

    # Saving and loading the model
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save_model(model, tokenizer, args)
        output_dir = args.output_dir
    else:
        output_dir = args.bert_model
    model, tokenizer = load_model(output_dir, args.do_lower_case, num_labels)

    # Evaluation
    model.to(device)
    evaluation(model, eval_examples, label_list, tokenizer, output_mode, device, num_labels, metric, args)