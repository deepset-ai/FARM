import logging

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


def create_examples_gnad(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = " ".join(line[1:])
        text_b = ""
        label = line[0]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


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
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
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
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples
