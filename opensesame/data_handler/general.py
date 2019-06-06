import csv
import sys
import logging
import torch
from torch.utils.data import (DataLoader,
                              TensorDataset)


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


def get_data_loader(examples, label_list, sampler, batch_size, max_seq_length, tokenizer, output_mode):
    # TODO: This should be a function that takes a processor and returns a dataloader
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = featurize_samples(examples,
                                                                                      label_list,
                                                                                      max_seq_length,
                                                                                      tokenizer,
                                                                                      output_mode)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    data_sampler = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return data_loader, dataset


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
