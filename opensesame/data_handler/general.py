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
    elif output_mode == "ner":
        from opensesame.data_handler import ner
        features = ner.convert_examples_to_features(samples, label_list, max_seq_length, tokenizer)
    else:
        raise Exception

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


def covert_features_to_dataset(features, label_dtype=torch.long):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=label_dtype)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
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
             local_rank=-1):
        db = cls(data_processor=data_processor,
                 examples_to_features_fn=convert_examples_to_features,
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
        feature_objects = self.examples_to_features(example_objects,
                                                    self.label_list, max_seq_len,
                                                    self.tokenizer,
                                                    output_mode="classification")
        dataset = self.features_to_dataset(feature_objects)
        labels = [x[3].item() for x in dataset]
        class_weights = list(compute_class_weight("balanced",
                                            np.unique(labels),
                                            labels))
        self.class_weights[dataset_name] = class_weights

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
        return self.class_weights[dataset]

    def get_data_loader(self, dataset):
        return self.loaders[dataset]

    def n_samples(self, dataset):
        return self.counts[dataset]

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


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

