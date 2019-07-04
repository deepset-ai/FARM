import os

import torch

from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.input_example import (
    create_examples_gnad,
    create_examples_conll_03,
    create_examples_germ_eval_18_coarse,
    create_examples_germ_eval_18_fine,
)
from farm.data_handler.input_features import (
    examples_to_features_sequence,
    examples_to_features_ner,
)
from farm.data_handler.utils import read_tsv, read_ner_file


class PreprocessingPipeline:
    """ Contains the pipeline of preprocessing functions that is used to turn a dataset from files into
    PyTorch Dataset objects. Also contains dataset specific information such as the list of labels, the
    evaluation metric and the type of task that it is. It is called most often by the DataBunch object. """

    def __init__(
        self,
        file_to_list,
        list_to_examples,
        examples_to_features,
        features_to_dataset,
        tokenizer,
        max_seq_len,
        label_list,
        metric,
        output_mode,
        token_level,
        filenames,
        dev_split,
        data_dir,
        delimiter="\t",
        label_dtype=torch.long,
    ):
        """ Creates a PreprocessingPipeline object with all the functions and information needed to handle a dataset."""

        assert len(filenames) == 3
        assert dev_split >= 0.0 and dev_split < 1.0

        # TODO RENAME PIPELINE STEPS
        # The main steps of the pipeline
        self.stages = ["file", "list", "examples", "features", "dataset"]
        self.pipeline = [
            self.call_file_to_list,
            self.call_list_to_examples,
            self.call_example_to_features,
            self.call_features_to_dataset,
        ]

        self.file_to_list = file_to_list
        self.list_to_examples = list_to_examples
        self.examples_to_features = examples_to_features
        self.features_to_dataset = features_to_dataset

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        self.label_dtype = label_dtype
        self.label_list = label_list
        self.metric = metric
        self.output_mode = output_mode
        self.token_level = token_level
        self.train_file = os.path.join(data_dir, filenames[0])
        self.test_file = os.path.join(data_dir, filenames[2])

        # TODO make sure dev split and dev file have compatible values
        try:
            self.dev_file = os.path.join(data_dir, filenames[1])
        except TypeError:
            self.dev_file = None
        self.dev_split = dev_split

    def convert(self, data, start, end):
        """ Applies a specified set of preprocessing pipeline steps to the data. """
        assert start in self.stages
        assert end in self.stages

        start_idx = self.stages.index(start)
        end_idx = self.stages.index(end)
        for i in range(start_idx, end_idx):
            pipeline_step = self.pipeline[i]
            data = pipeline_step(data)
        return data

    def call_file_to_list(self, filename):
        return self.file_to_list(filename=filename, delimiter=self.delimiter)

    def call_list_to_examples(self, data):
        return self.list_to_examples(lines=data, set_type="X")

    def call_example_to_features(self, data):
        return self.examples_to_features(
            examples=data,
            label_list=self.label_list,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )

    def call_features_to_dataset(self, data):
        return self.features_to_dataset(features=data, label_dtype=self.label_dtype)


class PPGNAD(PreprocessingPipeline):
    """ Used to handle the GNAD dataset (https://tblock.github.io/10kGNAD/)"""

    def __init__(self, data_dir, tokenizer, max_seq_len):

        # TODO how best to format this
        label_list = [
            "Web",
            "Sport",
            "International",
            "Panorama",
            "Wissenschaft",
            "Wirtschaft",
            "Kultur",
            "Etat",
            "Inland",
        ]
        metric = "acc"
        output_mode = "classification"
        token_level = False
        train_file = "train.csv"
        dev_file = None
        test_file = "test.csv"
        dev_split = 0.1

        super(PPGNAD, self).__init__(
            file_to_list=read_tsv,
            list_to_examples=create_examples_gnad,
            examples_to_features=examples_to_features_sequence,
            features_to_dataset=convert_features_to_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            delimiter=";",
            label_dtype=torch.long,
            metric=metric,
            output_mode=output_mode,
            token_level=token_level,
            filenames=[train_file, dev_file, test_file],
            dev_split=dev_split,
            data_dir=data_dir,
        )


class PPCONLL03(PreprocessingPipeline):
    """ Used to handle the CoNLL 2003 dataset (https://www.clips.uantwerpen.be/conll2003/ner/)"""

    def __init__(self, data_dir, tokenizer, max_seq_len):

        # TODO how best to format this
        label_list = [
            "[PAD]",
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "X",
            "B-OTH",
            "I-OTH",
            "[CLS]",
            "[SEP]",
        ]
        metric = "seq_f1"
        output_mode = "classification"
        token_level = True
        train_file = "train.txt"
        dev_file = "valid.txt"
        test_file = "test.txt"
        dev_split = 0.0

        super(PPCONLL03, self).__init__(
            file_to_list=read_ner_file,
            list_to_examples=create_examples_conll_03,
            examples_to_features=examples_to_features_ner,
            features_to_dataset=convert_features_to_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            label_dtype=torch.long,
            metric=metric,
            output_mode=output_mode,
            token_level=token_level,
            filenames=[train_file, dev_file, test_file],
            dev_split=dev_split,
            data_dir=data_dir,
        )


class PPGermEval18Fine(PreprocessingPipeline):
    """ """

    def __init__(self, data_dir, tokenizer, max_seq_len):

        # TODO how best to format this
        label_list = ['OTHER',
                      'INSULT',
                      'PROFANITY',
                      'ABUSE']
        metric = "f1_macro"
        output_mode = "classification"
        token_level = False
        train_file = "train.tsv"
        dev_file = None
        test_file = "test.tsv"
        dev_split = 0.1
        delimiter = "\t"

        super(PPGermEval18Fine, self).__init__(
            file_to_list=read_tsv,
            list_to_examples=create_examples_germ_eval_18_fine,
            examples_to_features=examples_to_features_sequence,
            features_to_dataset=convert_features_to_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            delimiter=delimiter,
            label_dtype=torch.long,
            metric=metric,
            output_mode=output_mode,
            token_level=token_level,
            filenames=[train_file, dev_file, test_file],
            dev_split=dev_split,
            data_dir=data_dir,
        )


class PPGermEval18Coarse(PreprocessingPipeline):
    """ """

    def __init__(self, data_dir, tokenizer, max_seq_len):

        # TODO how best to format this
        label_list = ['OTHER',
                      'OFFENSE']
        metric = "f1_macro"
        output_mode = "classification"
        token_level = False
        train_file = "train.tsv"
        dev_file = None
        test_file = "test.tsv"
        dev_split = 0.1
        delimiter = "\t"

        super(PPGermEval18Coarse, self).__init__(
            file_to_list=read_tsv,
            list_to_examples=create_examples_germ_eval_18_coarse,
            examples_to_features=examples_to_features_sequence,
            features_to_dataset=convert_features_to_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            delimiter=delimiter,
            label_dtype=torch.long,
            metric=metric,
            output_mode=output_mode,
            token_level=token_level,
            filenames=[train_file, dev_file, test_file],
            dev_split=dev_split,
            data_dir=data_dir,
        )


class PPGermEval14(PreprocessingPipeline):
    """ """

    def __init__(self, data_dir, tokenizer, max_seq_len):

        # TODO how best to format this
        label_list = [
            "[PAD]",
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "X",
            "B-OTH",
            "I-OTH",
            "[CLS]",
            "[SEP]",
        ]
        metric = "seq_f1"
        output_mode = "classification"
        token_level = True
        train_file = "train.txt"
        dev_file = "valid.txt"
        test_file = "test.txt"
        dev_split = 0.0

        super(PPGermEval14, self).__init__(
            file_to_list=read_ner_file,
            list_to_examples= create_examples_conll_03,
            examples_to_features=examples_to_features_ner,
            features_to_dataset=convert_features_to_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            label_dtype=torch.long,
            metric=metric,
            output_mode=output_mode,
            token_level=token_level,
            filenames=[train_file, dev_file, test_file],
            dev_split=dev_split,
            data_dir=data_dir,
        )
