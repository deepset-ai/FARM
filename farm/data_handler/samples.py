from farm.data_handler.utils import get_sentence_pair

import logging

logger = logging.getLogger(__name__)


class SampleBasket:
    def __init__(self,
                 id,
                 raw,
                 samples=None):
        self.id = id
        self.raw = raw
        self.samples = samples



class Sample(object):
    """A single training/test example."""

    def __init__(self,
                 id,
                 clear_text,
                 features=None):

        self.id = id
        # "train - 1 - 1
        self.clear_text = clear_text
        self.features = features


def create_examples_germ_eval_18_coarse(lines, set_type, text_a_index, label_index, text_b_index=None):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        # Skips the invalid example from the header of the file
        if label == "label":
            continue
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=None, label=label)
        )
    return examples


# def GNADInputExamples(Sample):
#
#     def featurize()
#         # currently examples to features
#
#
#     def create()
#         # currentlz create_samples

def create_samples_gnad(baskets, set_type):
    """Creates examples for the training and dev sets."""
    for (i, basket) in enumerate(baskets):
        id = "%s-%s" % (set_type, i)
        text = " ".join(basket.raw[1:])
        label = basket.raw[0]
        baskets.samples.append(
            Sample(id=id, clear_text={"text": text,
                                      "label": label} ))
    return baskets

def create_samples_one_label_one_text(raw_data, text_index, label_index, basket_id):

    text = " ".join(raw_data[text_index:])
    label = raw_data[label_index]

    return [Sample(id=basket_id + " - 1",
                    clear_text={"text": text,
                                "label": label})]

def create_examples_germ_eval_18_coarse(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        # Skips the invalid example from the header of the file
        if label == "label":
            continue
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=None, label=label)
        )
    return examples


def create_examples_germ_eval_18_fine(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[2]
        # Skips the invalid example from the header of the file
        if label == "label":
            continue
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=None, label=label)
        )
    return examples


def create_examples_conll_03(lines, set_type):
    examples = []
    for i, (sentence, label) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = " ".join(sentence)
        text_b = None
        label = label
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_mrpc(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[3]
        text_b = line[4]
        label = line[0]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_mnli(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[8]
        text_b = line[9]
        label = line[-1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_cola(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = line[3]
        label = line[1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=None, label=label)
        )
    return examples


def create_examples_sst2(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=None, label=label)
        )
    return examples


def create_examples_stsb(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[7]
        text_b = line[8]
        label = line[-1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_qqp(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        try:
            text_a = line[3]
            text_b = line[4]
            label = line[5]
        except IndexError:
            continue
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_qnli(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[1]
        text_b = line[2]
        label = line[-1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_rte(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[1]
        text_b = line[2]
        label = line[-1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


def create_examples_wnli(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[1]
        text_b = line[2]
        label = line[-1]
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


# TODO naming here would be better docs than lines. Just changed for temporary bug fixing
def create_examples_lm(lines, set_type):
    """Creates examples for Language Model Finetuning that consist of two sentences and the isNext label indicating if
     the two are subsequent sentences from one doc"""
    docs = lines[0]
    sample_to_docs = lines[1]
    examples = []
    for idx in range(len(sample_to_docs)):
        guid = "%s-%s" % (set_type, idx)
        text_a, text_b, is_next_label = get_sentence_pair(docs, sample_to_docs, idx)
        examples.append(
            Sample(guid=guid, text_a=text_a, text_b=text_b, label=is_next_label)
        )
    return examples
