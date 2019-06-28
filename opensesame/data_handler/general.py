import csv
import sys
import logging
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, DistributedSampler, TensorDataset)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, initial_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.initial_mask = initial_mask

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


def covert_features_to_dataset(features, label_dtype=torch.long):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=label_dtype)
    try:
        all_initial_masks = torch.tensor([f.initial_mask for f in features], dtype=torch.long)
    except TypeError:
        all_initial_masks = torch.tensor([0] * len(features), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_initial_masks)
    return dataset

def covert_dataset_to_dataloader(dataset, sampler, batch_size):
    sampler_initialized = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler_initialized, batch_size=batch_size)
    return data_loader

class NewDataBunch(object):
    # TODO BC: Currently I like this structure for the DataBunch but not sure about argument passing etc
    # One weakness of this functional approach is that function arguments are fixed and hard to refer back to original definition
    def __init__(self,
                 data_processor,
                 examples_to_features_fn,
                 tokenizer):

        self.data_processor = data_processor
        self.pairs_to_examples = data_processor._create_examples
        self.examples_to_features = examples_to_features_fn
        self.features_to_dataset = covert_features_to_dataset
        self.dataset_to_dataloader = covert_dataset_to_dataloader

        self.loaders = {}
        self.counts = {}
        self.class_weights = {}
        self.tokenizer = tokenizer
        self.label_list = self.data_processor.get_labels()
        self.num_labels = len(self.label_list)


    @classmethod
    def load(cls,
             data_dir,
             data_processor,
             tokenizer,
             batch_size,
             max_seq_len,
             examples_to_features_fn,
             local_rank=-1):
        """ TODO: NEED TO THINK OF THE RIGHT WAY TO CHOOSE CONVERT FEATURES FN"""
        db = cls(data_processor=data_processor,
                 examples_to_features_fn=examples_to_features_fn,
                 tokenizer=tokenizer)
        if local_rank == -1:
            train_sampler = RandomSampler
        else:
            train_sampler = DistributedSampler
        db.init_data(data_dir, "train", train_sampler, batch_size, max_seq_len)
        db.init_data(data_dir, "dev", SequentialSampler, batch_size, max_seq_len)
        db.init_data(data_dir, "test", SequentialSampler, batch_size, max_seq_len)
        return db

    @classmethod
    def load_inf(cls, processor, tokenizer, batch_size, max_seq_len):
        db = cls(processor, tokenizer, batch_size, max_seq_len)
        return db

    def init_data(self, dir, dataset_name, sampler, batch_size, max_seq_len):

        # TODO: I currently don't like that you have to tweak this and also the examples_to_features function to
        # adapt to your dataset. These two parts should be coupled together
        if dataset_name == "train":
            example_objects = self.data_processor.get_train_examples(dir)
        elif dataset_name == "dev":
            example_objects = self.data_processor.get_dev_examples(dir)
        elif dataset_name == "test":
            example_objects = self.data_processor.get_test_examples(dir)
        else:
            raise Exception

        self.counts[dataset_name] = len(example_objects)
        logger.info("Number of Loaded Samples: {}".format(self.counts[dataset_name]))
        feature_objects = self.examples_to_features(example_objects,
                                                    self.label_list, max_seq_len,
                                                    self.tokenizer,
                                                    output_mode="classification")
        dataset = self.features_to_dataset(feature_objects)

        try:
            labels = [x[3].item() for x in dataset]
            class_weights = list(compute_class_weight("balanced",
                                                np.unique(labels),
                                                labels))
            self.class_weights[dataset_name] = class_weights
        except ValueError:
            logger.info("Class weighting not available for token level tasks such as NER")

        data_loader = self.dataset_to_dataloader(dataset,
                                                 sampler,
                                                 batch_size)
        self.loaders[dataset_name] = data_loader

    def inference(self, input_sentences):
        label_text_pairs = [(";", sent) for sent in input_sentences]
        example_objects = self.pairs_to_examples(label_text_pairs)
        feature_objects = self.examples_to_features(example_objects)
        dataset = self.wrap_in_dataset(feature_objects)
        return self.wrap_in_dataloader(dataset, SequentialSampler)

    def get_class_weights(self, dataset):
        try:
            return self.class_weights[dataset]
        except KeyError:
            logger.warning("Class weights not available for {} dataset. This is perhaps due to the "
                           "fact that class weighting is currently not implemented for token level tasks like NER")

    def get_data_loader(self, dataset):
        return self.loaders[dataset]

    def n_samples(self, dataset):
        return self.counts[dataset]