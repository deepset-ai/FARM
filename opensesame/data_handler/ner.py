import os
import logging

from opensesame.data_handler.general import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


class ConllProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["[PAD]","O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "B-OTH", "I-OTH",
                "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)


    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token="[CLS]",
                                 pad_token="[PAD]",
                                 sep_token="[SEP]",
                                 non_initial_token="X",
                                 **kwargs):

    feature_objects = []

    for idx, example in enumerate(examples):
        # Tokenize words and extend the labels so they are aligned with the tokens
        words = example.text_a.split(" ")
        tokens, initial_mask = words_to_tokens(words, tokenizer, max_seq_length)

        labels_word = example.label
        labels_token = expand_labels(labels_word, initial_mask, non_initial_token)

        # Add CLS and SEP tokens
        tokens = add_cls_sep(tokens, cls_token, sep_token)
        labels_token = add_cls_sep(labels_token, cls_token, sep_token)
        initial_mask = [0] + initial_mask + [0]      # CLS and SEP don't count as initial tokens
        input_mask = [1] * len(tokens)

        # Convert to input and labels to ids, generate masks
        # Todo: Something is odd here because [PAD] is index one in the vocab of tokenizer but we are padding with 0, or maybe it doesnt matter because its masked out anyways
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_list.index(lt) for lt in labels_token]
        segment_ids = [0] * max_seq_length

        # Pad
        input_ids = pad(input_ids, max_seq_length, 0)
        label_ids = pad(label_ids, max_seq_length, 0)
        initial_mask = pad(initial_mask, max_seq_length, 0)
        input_mask = pad(input_mask, max_seq_length, 0)


        if idx < 5:
            print_example_with_features(example,
                                        tokens,
                                        input_ids,
                                        input_mask,
                                        segment_ids,
                                        label_ids,
                                        initial_mask)

        feature_object = InputFeatures(input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_id=label_ids,
                                       initial_mask=initial_mask)
        feature_objects.append(feature_object)

    return feature_objects


def print_example_with_features(example,
                                tokens,
                                input_ids,
                                input_mask,
                                segment_ids,
                                label_ids,
                                initial_mask):
    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("tokens: %s" % " ".join(
        [str(x) for x in tokens]))
    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    logger.info(
        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    logger.info("label: %s" % (example.label))
    logger.info("ids  : %s" % (label_ids))
    logger.info("initial_mask: %s" % (initial_mask))


def add_cls_sep(seq, cls_token, sep_token):
    ret = [cls_token]
    ret += seq
    ret += [sep_token]
    return ret


def pad(seq, max_seq_len, pad_token):
    ret = seq
    n_required_pad = max_seq_len - len(seq)
    for _ in range(n_required_pad):
        ret.append(pad_token)
    return ret


def expand_labels(labels_word,
                  initial_mask,
                  non_initial_token):
    labels_token = []
    word_index = 0
    for im in initial_mask:
        if im:
            # i.e. if token is word initial
            labels_token.append(labels_word[word_index])
            word_index += 1
        else:
            # i.e. token is not the first in the word
            labels_token.append(non_initial_token)

    assert len(labels_token) == len(initial_mask)
    return labels_token


def words_to_tokens(words, tokenizer, max_seq_length):
    tokens_all = []
    initial_mask = []
    for w in words:
        tokens_word = tokenizer.tokenize(w)

        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue

        n_non_initial_tokens = len(tokens_word) - 1
        initial_mask += [1]
        for _ in range(n_non_initial_tokens):
            initial_mask += [0]
        tokens_all += tokens_word

    # Clip at max_seq_length. The "-2" is for CLS and SEP token
    tokens_all = tokens_all[:max_seq_length - 2]
    initial_mask = initial_mask[:max_seq_length - 2]

    assert len(tokens_all) == len(initial_mask)

    return tokens_all, initial_mask