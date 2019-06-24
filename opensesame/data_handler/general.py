import csv
import sys
import logging
import torch
from torch.utils.data import (DataLoader,RandomSampler, SequentialSampler, DistributedSampler, TensorDataset)


logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def featurize_samples(samples, label_list, max_seq_length, tokenizer, output_mode):
    """ Convert examples to proper tensors for TensorDataset"""
    if output_mode in ("classification", "regression"):
        from opensesame.data_handler import seq_classification
        features = seq_classification.convert_examples_to_features(
            samples, label_list, max_seq_length, tokenizer, output_mode)
    if output_mode == "ner":
        from opensesame.data_handler import ner
        features = ner.convert_examples_to_features(samples, label_list, max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode in ("ner","classification"):
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        raise NotImplementedError
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

def get_dataset(examples, label_list, tokenizer, max_seq_length, output_mode):
    # TODO: This should be a function that takes a processor and returns a dataloader
    logger.info("  Num examples = %d", len(examples))
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = featurize_samples(examples,
                                                                                      label_list,
                                                                                      max_seq_length,
                                                                                      tokenizer,
                                                                                      output_mode)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BertDataBunch(object):

    def __init__(self, data_dir, processor, output_mode, tokenizer, train_batch_size, max_seq_length, local_rank=-1):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.batch_size = train_batch_size
        self.local_rank = local_rank
        self.output_mode = output_mode # TODO: this is bad naming. we should switch to prediction_head or downstream_task?
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.test_is_dev = False

        # init all dataloaders
        self.init_train_data()
        self.init_dev_data()
        self.init_test_data()

    def init_train_data(self):
        logger.info("***** Loading train data ******")
        train_examples = self.processor.get_train_examples(self.data_dir)
        self.num_train_examples = len(train_examples)

        if self.local_rank == -1:
            train_sampler = RandomSampler
        else:
            train_sampler = DistributedSampler

        self.train_dataset = get_dataset(train_examples, self.label_list, self.tokenizer, self.max_seq_length, self.output_mode)
        train_sampler = train_sampler(self.train_dataset)
        self.train_data_loader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)


    def init_dev_data(self):
        logger.info("***** Loading dev data ******")
        dev_examples = self.processor.get_dev_examples(self.data_dir)
        self.num_dev_evamples = len(dev_examples)

        dev_sampler = SequentialSampler

        self.dev_dataset = get_dataset(dev_examples, self.label_list, self.tokenizer, self.max_seq_length, self.output_mode)
        dev_sampler = dev_sampler(self.dev_dataset)
        self.dev_data_loader = DataLoader(self.dev_dataset, sampler=dev_sampler, batch_size=self.batch_size)

    def init_test_data(self):
        logger.info("***** Loading test data ******")
        try:
            test_examples = self.processor.get_test_examples(self.data_dir)
            self.num_test_examples = len(test_examples)

            test_sampler = SequentialSampler

            self.test_dataset = get_dataset(test_examples, self.label_list, self.tokenizer, self.max_seq_length,
                                           self.output_mode)
            test_sampler = test_sampler(self.test_dataset)
            self.test_data_loader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.batch_size)

        except:
            logger.warning(
                "Test set not found, evaluation during training and afterwards will both be performed on dev set.")
            self.test_data_loader = self.dev_data_loader
            self.num_test_examples = self.num_dev_examples
            self.test_is_dev = True

