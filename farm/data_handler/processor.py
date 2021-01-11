import abc
import inspect
import json
import logging
import os
import random
from abc import ABC
from inspect import signature
from pathlib import Path
from random import randint

import numpy as np
import torch
from numpy.random import random as random_float
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig
from tokenizers import Encoding

from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.input_features import get_roberta_seq_2_start, get_camembert_seq_2_start
from farm.data_handler.input_features import sample_to_features_text
from farm.data_handler.nq_utils import (
    sample_to_features_qa_Natural_Questions,
    create_samples_qa_Natural_Question,
    convert_qa_input_dict,
)

from farm.data_handler.samples import (
    Sample,
    SampleBasket,
    get_passage_offsets,
    offset_to_token_idx_vecorized
)

from farm.data_handler.utils import (
    pad,
    expand_labels,
    read_tsv,
    read_tsv_sentence_pair,
    read_docs_from_txt,
    read_ner_file,
    read_squad_file,
    read_jsonl,
    read_dpr_json,
    is_json,
    get_sentence_pair,
    split_with_metadata,
)
from farm.modeling.tokenization import (
    Tokenizer,
    tokenize_with_metadata,
    truncate_sequences,
    tokenize_batch_question_answering,
    _get_start_of_word
)
from farm.utils import MLFlowLogger as MlLogger
from farm.utils import try_get

from tokenizers.pre_tokenizers import WhitespaceSplit

ID_NAMES = ["example_id", "external_id", "doc_id", "id"]


logger = logging.getLogger(__name__)


class Processor(ABC):
    """
    Is used to generate PyTorch Datasets from input data. An implementation of this abstract class should be created
    for each new data source.
    Implement the abstract methods: file_to_dicts(), _dict_to_samples(), _sample_to_features()
    to be compatible with your data format
    """

    subclasses = {}

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        train_filename,
        dev_filename,
        test_filename,
        dev_split,
        data_dir,
        tasks={},
        proxies=None,
        multithreading_rust=True,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :type data_dir: str
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :type tasks: dict
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param multithreading_rust: Whether to allow multithreading in Rust, e.g. for FastTokenizers.
                                    Note: Enabling multithreading in Rust AND multiprocessing in python might cause
                                    deadlocks.
        :type multithreading_rust: bool
        """
        if not multithreading_rust:
            os.environ["RAYON_RS_NUM_CPUS"] = "1"

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None
        self.baskets = []

        self._log_params()
        self.problematic_sample_ids = set()

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(
        cls,
        processor_name,
        data_dir,
        tokenizer,
        max_seq_len,
        train_filename,
        dev_filename,
        test_filename,
        dev_split,
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :type processor_name: str
        :param data_dir: Directory where data files are located.
        :type data_dir: str
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced.
                          Only works if dev_filename is set to None
        :type dev_split: float
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: An instance of the specified processor.
        """

        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(
            f"Got more parameters than needed for loading {processor_name}: {unused_args}. "
            f"Those won't be used!"
        )
        processor = cls.subclasses[processor_name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            **kwargs,
        )

        return processor

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Infers the specific type of Processor from a config file (e.g. GNADProcessor) and loads an instance of it.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. GNADProcessor)
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        config["inference"] = True
        # init tokenizer
        if "lower_case" in config.keys():
            logger.warning("Loading tokenizer from deprecated FARM config. "
                           "If you used `custom_vocab` or `never_split_chars`, this won't work anymore.")
            tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"], do_lower_case=config["lower_case"])
        else:
            tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name,
                               metric=task["metric"],
                               label_list=task["label_list"],
                               label_column_name=task["label_column_name"],
                               text_column_name=task.get("text_column_name", None),
                               task_type=task["task_type"])

        if processor is None:
            raise Exception

        return processor

    @classmethod
    def convert_from_transformers(cls, tokenizer_name_or_path, task_type, max_seq_len, doc_stride,
                                  revision=None, tokenizer_class=None, tokenizer_args=None, use_fast=True):
        config = AutoConfig.from_pretrained(tokenizer_name_or_path, revision=revision)
        tokenizer_args = tokenizer_args or {}
        tokenizer = Tokenizer.load(tokenizer_name_or_path,
                                   tokenizer_class=tokenizer_class,
                                   use_fast=use_fast,
                                   revision=revision,
                                   **tokenizer_args,
                                   )

        # TODO infer task_type automatically from config (if possible)
        if task_type == "question_answering":
            processor = SquadProcessor(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                label_list=["start_token", "end_token"],
                metric="squad",
                data_dir="data",
                doc_stride=doc_stride
            )
        elif task_type == "embeddings":
            processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)

        elif task_type == "text_classification":
            label_list = list(config.id2label[id] for id in range(len(config.id2label)))
            processor = TextClassificationProcessor(tokenizer=tokenizer,
                                                    max_seq_len=max_seq_len,
                                                    data_dir="data",
                                                    label_list=label_list,
                                                    label_column_name="label",
                                                    metric="acc",
                                                    quote_char='"',
                                                    )
        elif task_type == "ner":
            label_list = list(config.id2label.values())
            processor = NERProcessor(
                tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir="data", metric="seq_f1",
                label_list=label_list
            )
        else:
            raise ValueError(f"`task_type` {task_type} is not supported yet. "
                             f"Valid options for arg `task_type`: 'question_answering', "
                             f"'embeddings', 'text_classification', 'ner'")

        return processor

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        # self.__dict__ doesn't give parent class attributes
        for key, value in inspect.getmembers(self):
            if is_json(value) and key[0] != "_":
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    def add_task(self, name,  metric, label_list, label_column_name=None,
                 label_name=None, task_type=None, text_column_name=None):
        if type(label_list) is not list:
            raise ValueError(f"Argument `label_list` must be of type list. Got: f{type(label_list)}")

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type
        }

    @abc.abstractmethod
    def file_to_dicts(self, file: str) -> [dict]:
        raise NotImplementedError()

    def _dict_to_samples(cls, dictionary: dict, all_dicts=None) -> [Sample]:
        raise NotImplementedError()

    def _sample_to_features(cls, sample: Sample) -> dict:
        raise NotImplementedError()

    def _dict_to_samples_and_features(self, dictionary: dict, all_dicts=None) -> [Sample]:
        raise NotImplementedError()

    def _init_samples_in_baskets(self):
        all_dicts = [b.raw for b in self.baskets]
        for basket in self.baskets:
            try:
                basket.samples = self._dict_to_samples(dictionary=basket.raw, all_dicts=all_dicts)
                for num, sample in enumerate(basket.samples):
                     sample.id = f"{basket.id_internal}-{num}"
            except Exception as e:
                logger.error(f"Could not create sample(s) from this dict: \n {basket.raw}")
                logger.error(f"Error message: {e}")

    def _featurize_samples(self):
        curr_problematic_sample_ids = []
        for basket in self.baskets:
            for sample in basket.samples:
                try:
                    sample.features = self._sample_to_features(sample=sample)
                except Exception as e:
                    curr_problematic_sample_ids.append(sample.id)
        if curr_problematic_sample_ids:
            self.problematic_sample_ids.update(curr_problematic_sample_ids)

    @staticmethod
    def log_problematic(problematic_sample_ids):
        if problematic_sample_ids:
            n_problematic = len(problematic_sample_ids)
            problematic_id_str = ", ".join(problematic_sample_ids)
            logger.error(
                f"Unable to convert {n_problematic} samples to features. Their ids are : {problematic_id_str}")


    def _init_and_featurize_samples_in_baskets(self):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
            try:
                basket.samples = self._dict_to_samples_and_features(dictionary=basket.raw,
                                                                    all_dicts=all_dicts,
                                                                    basket_id_internal=basket.id_internal)
                for num, sample in enumerate(basket.samples):
                    sample.id = f"{basket.id_internal}-{num}"
            except Exception as e:
                logger.error(f"Could not create sample(s) from this dict: \n {basket.raw}")
                logger.error(f"Error message: {e}")


    @staticmethod
    def _check_sample_features(basket):
        """Check if all samples in the basket has computed its features.

        Args:
            basket: the basket containing the samples

        Returns:
            True if all the samples in the basket has computed its features, False otherwise

        """
        if len(basket.samples) == 0:
            return False
        for sample in basket.samples:
            if sample.features is None:
                return False
        return True

    def _create_dataset(self):
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                self.baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names

    def dataset_from_dicts(self, dicts, indices=None, return_baskets = False):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This can be used for inference mode.

        :param dicts: List of dictionaries where each contains the data of one input sample.
        :type dicts: list of dicts
        :return: a Pytorch dataset and a list of tensor names.
        """
        # We need to add the index (coming from multiprocessing chunks) to have a unique basket ID

        self.baskets = []
        for id_internal, d in enumerate(dicts):
            id_external = self._id_from_dict(d)
            if indices:
                id_internal = indices[id_internal]
            self.baskets.append(SampleBasket(raw=d, id_external=id_external, id_internal=id_internal))

        self._init_samples_in_baskets()
        self._featurize_samples()
        if indices:
            if 0 in indices:
                self._log_samples(1)
        else:
            self._log_samples(1)

        dataset, tensor_names = self._create_dataset()
        # This mode is for inference where we need to keep baskets
        if return_baskets:
            #TODO simplify
            return dataset, tensor_names, self.problematic_sample_ids, self.baskets
        # This mode is for training where we can free ram by removing baskets
        else:
            return dataset, tensor_names, self.problematic_sample_ids

    def _log_samples(self, n_samples):
        logger.info("*** Show {} random examples ***".format(n_samples))
        for i in range(n_samples):
            random_basket = random.choice(self.baskets)
            random_sample = random.choice(random_basket.samples)
            logger.info(random_sample)

    def _log_params(self):
        params = {
            "processor": self.__class__.__name__,
            "tokenizer": self.tokenizer.__class__.__name__,
        }
        names = ["max_seq_len", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        MlLogger.log_params(params)

    @staticmethod
    def _id_from_dict(d):
        ext_id = try_get(ID_NAMES, d)
        if not ext_id and "qas" in d:
            ext_id = try_get(ID_NAMES, d["qas"][0])
        return ext_id


#########################################
# Processors for Text Classification ####
#########################################
class TextClassificationProcessor(Processor):
    """
    Used to handle the text classification datasets that come in tabular format (CSV, TSV, etc.)
    """
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param multilabel: set to True for multilabel classification
        :type multilabel: bool
        :param header: which line to use as a header in the input csv/tsv
        :type  header: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        #TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,

        )
        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(name="text_classification",
                          metric=metric,
                          label_list=label_list,
                          label_column_name=label_column_name,
                          text_column_name=text_column_name,
                          task_type=task_type)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {}
        for task in self.tasks.values():
            column_mapping[task["label_column_name"]] = task["label_name"]
            column_mapping[task["text_column_name"]] = "text"
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            rename_columns=column_mapping,
            header=self.header,
            proxies=self.proxies,
            max_samples=self.max_samples
            )

        return dicts

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, debug=False):
        self.baskets = []
        # Tokenize in batches
        texts = [x["text"] for x in dicts]
        tokenized_batch = self.tokenizer.batch_encode_plus(
            texts,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length"
        )
        input_ids_batch = tokenized_batch["input_ids"]
        segment_ids_batch = tokenized_batch["token_type_ids"]
        padding_masks_batch = tokenized_batch["attention_mask"]

        # From here we operate on a per sample basis
        for dictionary, input_ids, segment_ids, padding_mask in zip(
                dicts, input_ids_batch, segment_ids_batch, padding_masks_batch
        ):

            # TODO Build tokenized dict for debug mode
            tokenized = {}

            feat_dict = {"input_ids": input_ids,
                         "padding_mask": padding_mask,
                         "segment_ids": segment_ids}

            # Create labels
            # i.e. not inference
            if not return_baskets:
                label_dict = self.generate_labels(dictionary)
                feat_dict.update(label_dict)

            # Add Basket to self.baskets
            curr_sample = Sample(id=None,
                                 clear_text=dictionary,
                                 tokenized=tokenized,
                                 features=[feat_dict])
            curr_basket = SampleBasket(id_internal=None,
                                       raw=dictionary,
                                       id_external=None,
                                       samples=[curr_sample])
            self.baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(1)

        # TODO populate problematic ids
        problematic_ids = set()
        logger.warning("Currently no support in TextClassification processor for returning problematic ids")
        dataset, tensornames = self._create_dataset()
        ret = [dataset, tensornames, problematic_ids]
        if return_baskets:
            ret.append(self.baskets)
        return ret

    def _create_dataset(self):
        # TODO this is the proposed new version to replace the mother function
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names

    def generate_labels(self, dictionary):
        ret = {}
        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            label_name = task["label_name"]
            label_raw = dictionary[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                label_ids = [label_list.index(label_raw)]
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            ret[task["label_tensor_name"]] = label_ids
        return ret


class TextPairClassificationProcessor(TextClassificationProcessor):
    """
    Used to handle text pair classification datasets (e.g. Answer Selection or Natural Inference) that come in
    tsv format. The columns should be called text, text_b and label.
    """
    def __init__(self, **kwargs):
        super(TextPairClassificationProcessor, self).__init__(**kwargs)

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
        dicts = read_tsv_sentence_pair(
            rename_columns=column_mapping,
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            proxies=self.proxies,
        )
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        tokenized_a = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        tokenized_b = tokenize_with_metadata(dictionary["text_b"], self.tokenizer)

        if len(tokenized_a["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []
        if len(tokenized_b["tokens"]) == 0:
            text_b = dictionary["text_b"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text_b}")
            return []

        tokenized = {"tokens": tokenized_a["tokens"],
                     "tokens_b": tokenized_b["tokens"]}
        tokenized["tokens"], tokenized["tokens_b"], _ = truncate_sequences(seq_a=tokenized["tokens"],
                                                                           seq_b=tokenized["tokens_b"],
                                                                           tokenizer=self.tokenizer,
                                                                           max_seq_len=self.max_seq_len)
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        seq_a = dictionary["text"]
        seq_b = dictionary["text_b"]

        inputs = self.tokenizer.encode_plus(
            text=seq_a,
            text_pair=seq_b,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

        # Find position of [SEP]-token
        # seq_2_start_t is the index of the first token in the second text sequence (e.g. passage)
        if "RobertaTokenizer" in self.tokenizer.__class__.__name__:
            seq_2_start_t = get_roberta_seq_2_start(input_ids)
        elif "CamembertTokenizer" in self.tokenizer.__class__.__name__:
            seq_2_start_t = get_camembert_seq_2_start(input_ids)
        else:
            seq_2_start_t = segment_ids.index(1)

        # Get tokens as text with metadata
        tokens_a = []
        tokens_b = []
        for idx, (token_id, is_special_token) in enumerate(zip(input_ids,
                                                               inputs["special_tokens_mask"])):
            if not is_special_token:
                if idx < seq_2_start_t:
                    tokens_a.append(self.tokenizer.convert_ids_to_tokens(token_id))
                else:
                    tokens_b.append(self.tokenizer.convert_ids_to_tokens(token_id))

        token_dict = {"tokens": tokens_a,
                      "tokens_b": tokens_b}

        if len(token_dict["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {seq_a}")
            return []
        if len(token_dict["tokens_b"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {seq_b}")
            return []

        # Build feature dict

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)
        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)

        assert len(input_ids) == self.max_seq_len
        assert len(padding_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        feat_dict = {"input_ids": input_ids,
                     "padding_mask": padding_mask,
                     "segment_ids": segment_ids}

        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            try:
                label_name = task["label_name"]
                label_raw = dictionary[label_name]
                label_list = task["label_list"]
                if task["task_type"] == "classification":
                    # id of label
                    try:
                        label_ids = [label_list.index(label_raw)]
                    except ValueError as e:
                        raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
                elif task["task_type"] == "multilabel_classification":
                    # multi-hot-format
                    label_ids = [0] * len(label_list)
                    for l in label_raw.split(","):
                        if l != "":
                            label_ids[label_list.index(l)] = 1
                elif task["task_type"] == "regression":
                    label_ids = [float(label_raw)]
                else:
                    raise ValueError(task["task_type"])
            except KeyError:
                # For inference mode we don't expect labels
                label_ids = None
            if label_ids is not None:
                feat_dict[task["label_tensor_name"]] = label_ids

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feat_dict])]




#########################################
# Processors for Basic Inference ####
#########################################
class InferenceProcessor(Processor):
    """
    Generic processor used at inference time:
    - fast
    - no labels
    - pure encoding of text into pytorch dataset
    - Doesn't read from file, but only consumes dictionaries (e.g. coming from API requests)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        **kwargs,
    ):

        super(InferenceProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=None,
            dev_filename=None,
            test_filename=None,
            dev_split=None,
            data_dir=None,
            tasks={},
        )

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Overwriting method from parent class to **always** load the InferenceProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an InferenceProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"])
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name="InferenceProcessor", **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def file_to_dicts(self, file: str) -> [dict]:
        raise NotImplementedError

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None, tokenizer=self.tokenizer,
                                                max_seq_len=self.max_seq_len)
        return Sample(id=None, clear_text=dictionary, tokenized=tokenized)

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, debug=False):
        self.baskets = []

        if not self.tokenizer.is_fast:
            for d in dicts:
                sample = self._dict_to_samples(dictionary=d)
                features = self._sample_to_features(sample)
                sample.features = features
                basket = SampleBasket(id_internal=None,
                                       raw=d,
                                       id_external=None,
                                       samples=[sample])
                self.baskets.append(basket)
        else:
            # Tokenize in batches
            texts = [x["text"] for x in dicts]
            tokenized_batch = self.tokenizer.batch_encode_plus(
                texts,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                truncation=True,
                max_length=self.max_seq_len,
                add_special_tokens=True,
                padding="max_length"
            )
            input_ids_batch = tokenized_batch["input_ids"]
            segment_ids_batch = tokenized_batch["token_type_ids"]
            padding_masks_batch = tokenized_batch["attention_mask"]
            if self.tokenizer.is_fast:
                tokens_batch = [x.tokens for x in tokenized_batch.encodings]
            special_tokens_mask_batch = tokenized_batch["special_tokens_mask"]

            # From here we operate on a per sample basis
            for dictionary, input_ids, segment_ids, padding_mask, tokens, special_tokens_mask in zip(
                    dicts, input_ids_batch, segment_ids_batch, padding_masks_batch, tokens_batch, special_tokens_mask_batch
            ):

                # TODO Build tokenized dict for debug mode
                tokenized = {"tokens": [t for t, stm in zip(tokens, special_tokens_mask) if not stm]}

                feat_dict = {"input_ids": input_ids,
                             "padding_mask": padding_mask,
                             "segment_ids": segment_ids}

                # Add Basket to self.baskets
                curr_sample = Sample(id=None,
                                     clear_text=dictionary,
                                     tokenized=tokenized,
                                     features=[feat_dict])
                basket = SampleBasket(id_internal=None,
                                           raw=dictionary,
                                           id_external=None,
                                           samples=[curr_sample])
                self.baskets.append(basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(1)

        # TODO populate problematic ids
        problematic_ids = set()
        logger.warning("Currently no support in InferenceProcessor for returning problematic ids")
        dataset, tensornames = self._create_dataset()
        ret = [dataset, tensornames, problematic_ids]
        if return_baskets:
            ret.append(self.baskets)
        return ret


#########################################
# Processors for NER data ####
#########################################
class NERProcessor(Processor):
    """
    Used to handle most NER datasets, like CoNLL or GermEval 2014
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.txt",
        dev_filename="dev.txt",
        test_filename="test.txt",
        dev_split=0.0,
        delimiter="\t",
        proxies=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings).
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "seq_f1".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["seq_f1", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file. German version of Conll03 uses a whitespace. GermEval 2014 is tab separated \t
        :type delimiter: str
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        # Custom processor attributes
        self.delimiter = delimiter

        self.pre_tokenizer = WhitespaceSplit()

        super(NERProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )
        if metric and label_list:
            self.add_task("ner", metric, label_list)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        dicts = read_ner_file(filename=file, sep=self.delimiter,  proxies=self.proxies)
        return dicts

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, non_initial_token="X"):
        self.baskets = []

        # Perform batch tokenization
        texts = [x["text"] for x in dicts]
        words_and_spans = [self.pre_tokenizer.pre_tokenize_str(x) for x in texts]
        words = [[x[0] for x in y] for y in words_and_spans]

        # word_spans_batch is the char span for each whitespace split word
        word_spans_batch = [[x[1] for x in y] for y in words_and_spans]

        tokenized_batch = self.tokenizer.batch_encode_plus(
            words,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            is_split_into_words=True
        )

        # Create features by iterating over samples
        for i in range(len(dicts)):
            tokenized = tokenized_batch[i]
            d = dicts[i]

            # Either try to extract an ID from the dictionary, or else create an id
            # based on the order of the dictionaries coming in, taking into account
            # the indices generated by chunking and multiprocessing
            id_external = self._id_from_dict(d)
            if indices:
                id_internal = indices[i]
            else:
                id_internal = i

            input_ids = tokenized.ids
            segment_ids = tokenized.type_ids

            # We construct a mask to identify the first token of a word. We will later only use them for predicting entities.
            # Special tokens don't count as initial tokens => we add 0 at the positions of special tokens
            # For BERT we add a 0 in the start and end (for CLS and SEP)
            initial_mask = self._get_start_of_word(tokenized.words)
            assert len(initial_mask) == len(input_ids)

            # This mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            padding_mask = tokenized.attention_mask

            # i.e. if inference, we need to populate the tokenized_dict so that formatted preds can align
            # the prediction to the original text
            if return_baskets:
                token_to_word_map = tokenized.words
                word_spans = word_spans_batch[i]
                tokenized_dict = {
                    "tokens": tokenized.tokens,
                    "word_spans": word_spans,
                    "token_to_word_map": token_to_word_map,
                    "start_of_word": initial_mask
                }
            else:
                tokenized_dict = {}

            feature_dict = {
                "input_ids": input_ids,
                "padding_mask": padding_mask,
                "segment_ids": segment_ids,
                "initial_mask": initial_mask,
            }

            for task_name, task in self.tasks.items():
                try:
                    label_list = task["label_list"]
                    label_name = task["label_name"]
                    label_tensor_name = task["label_tensor_name"]
                    labels_word = d[label_name]
                    labels_token = expand_labels(labels_word, initial_mask, non_initial_token)
                    label_ids = [label_list.index(lt) for lt in labels_token]
                except ValueError:
                    # Usually triggered if label is not in label list
                    label_ids = None
                    problematic_labels = set(labels_token).difference(set(label_list))
                    logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                                   f"\nWe found a problem with labels {str(problematic_labels)}")
                # TODO change this when inference flag is implemented
                except KeyError:
                    # Usually triggered if there is no label in the sample
                    # This is expected during inference since there are no labels
                    # During training, this is a problem
                    label_ids = None
                    logger.warning(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                                   "\nIf your are running in *inference* mode: Don't worry!"
                                   "\nIf you are running in *training* mode: Verify you are supplying a proper label list to your processor and check that labels in input data are correct.")

                if label_ids:
                    feature_dict[label_tensor_name] = label_ids

            curr_sample = Sample(id=None,
                                 clear_text=d,
                                 tokenized=tokenized_dict,
                                 features=[feature_dict])
            curr_basket = SampleBasket(id_internal=id_internal,
                                       raw=d,
                                       id_external=id_external,
                                       samples=[curr_sample])
            self.baskets.append(curr_basket)

        # Don't log if we are processing a dataset chunk other than the first chunk
        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(1)

        dataset, tensor_names = self._create_dataset()
        ret = [dataset, tensor_names, self.problematic_sample_ids]
        # This is for inference where we need to keep baskets
        # By contrast, in training, we can remove baskets to free up RAM
        if return_baskets:
            ret.append(self.baskets)
        return tuple(ret)

    @staticmethod
    def _get_start_of_word(word_ids):
        words = np.array(word_ids)
        words[words == None] = -1
        start_of_word_single = [0] + list(np.ediff1d(words) > 0)
        start_of_word_single = [int(x) for x in start_of_word_single]
        return start_of_word_single

#####################
# LM Processors ####
#####################
class BertStyleLMProcessor(Processor):
    """
    Prepares data for masked language model training and next sentence prediction in the style of BERT
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename="train.txt",
        dev_filename="dev.txt",
        test_filename="test.txt",
        dev_split=0.0,
        next_sent_pred=True,
        next_sent_pred_style="bert-style",
        max_docs=None,
        proxies=None,
        masked_lm_prob=0.15,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param next_sent_pred: Whether to use next_sentence_prediction objective or not
        :type next_sent_pred: bool
        :param next_sent_pred_style:
            Two different styles for next sentence prediction available:
                - "sentence":   Use of a single sentence for Sequence A and a single sentence for Sequence B
                - "bert-style": Fill up all of max_seq_len tokens and split into Sequence A and B at sentence border.
                                If there are too many tokens, Sequence B will be truncated.
        :type next_sent_pred_style: str
        :param max_docs: maximum number of documents to include from input dataset
        :type max_docs: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param masked_lm_prob: probability of masking a token
        :type masked_lm_prob: float
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        self.delimiter = ""
        self.max_docs = max_docs

        if not tokenizer.is_fast:
            raise ValueError("This processor only supports FastTokenizers. "
                             "Load one by calling Tokenizer.load(..., use_fast=True)")

        super(BertStyleLMProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )

        self.next_sent_pred = next_sent_pred
        self.next_sent_pred_style = next_sent_pred_style
        added_tokens = self.get_added_tokens()
        self.add_task("lm", "acc", list(self.tokenizer.vocab) + added_tokens)
        if self.next_sent_pred:
            self.add_task("nextsentence", "acc", ["False", "True"])
        self.masked_lm_prob = masked_lm_prob

    def get_added_tokens(self):
        dictionary = self.tokenizer.get_added_vocab()
        sorted_tuples = sorted(dictionary.items(), key=lambda x: x[0])
        return [x[1] for x in sorted_tuples]

    def file_to_dicts(self, file: str) -> list:
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter, max_docs=self.max_docs, proxies=self.proxies)
        return dicts

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        dicts = [d["doc"] for d in dicts]
        # 1) Create samples & truncate (sentence pairs)
        # next sentence prediction ...
        if self.next_sent_pred:
            assert len(dicts) > 1, "Need at least 2 documents to sample random sentences from"
            # ...with single sentences
            if self.next_sent_pred_style == "sentence":
                samples = self._create_sequence_pairs_by_line(dicts)
            # ...bert style
            elif self.next_sent_pred_style == "bert-style":
                 samples = self._create_sequence_pairs_bert_style(dicts)
            else:
                raise NotImplementedError("next_sent_pred_style has to be 'sentence' or 'bert-style'")

        # no next sentence prediction
        else:
           samples = self._create_sequence_pairs_no_next_sent(dicts)

        # 2) Create labels (masking words + NSP)
        features = []
        vocab_length = len(self.tokenizer.vocab)-1
        for sample in samples:
            features.append(self._create_labels(sample=sample, vocab_length=vocab_length))

        # 3) Create dataset
        dataset, tensor_names = convert_features_to_dataset(features=features)
        return dataset, tensor_names, set()

    def _create_sequence_pairs_by_line(self, docs):
        samples = []
        raw_pairs = []
        labels = []
        for doc in docs:
            # create one sample for each sentence in the doc (except for the very last -> "nextSentence" is impossible)
            for idx in range(len(doc) - 1):
                text_a, text_b, is_next_label = get_sentence_pair(doc, docs, idx)
                raw_pairs.append((text_a, text_b))
                labels.append(is_next_label)

        # Tokenize + Encode masks
        encoded_pairs = self.tokenizer.batch_encode_plus(raw_pairs,
                                                         max_length=self.max_seq_len,
                                                         truncation=True,
                                                         truncation_strategy="longest_first",
                                                         add_special_tokens=True,
                                                         padding='max_length'
                                                         )

        assert len(encoded_pairs.input_ids) == len(raw_pairs)

        # Create "Start of word mask"
        start_of_word = []
        for e in encoded_pairs.encodings:
            start_of_word.append(_get_start_of_word(e.words, e.special_tokens_mask))

        # Create Sample objects
        for idx in range(len(raw_pairs)):
            if len(encoded_pairs.input_ids[idx]) == 0:
                logger.warning(
                    f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {raw_pairs[idx]}")
                continue

            # We don't populate 'tokenized' here as we skiped the intermediate string token stage abeoce to improve the speed ...
            samples.append(Sample(id=None,
                                  clear_text={"text_a": raw_pairs[idx][0],
                                              "text_b": raw_pairs[idx][1],
                                              "nextsentence_label": labels[idx]},
                                  tokenized={"tokens": encoded_pairs.encodings[idx].tokens,
                                             "start_of_word": start_of_word[idx],
                                             "special_tokens_mask": encoded_pairs.encodings[idx].special_tokens_mask,
                                             "offsets": encoded_pairs.encodings[idx].offsets},
                                  features={"input_ids": encoded_pairs.input_ids[idx],
                                            "segment_ids": encoded_pairs.token_type_ids[idx],
                                            "padding_mask": encoded_pairs.attention_mask[idx],
                                            }
                                  ))
        return samples

    def _create_sequence_pairs_bert_style(self, docs):
        samples = []

        # 1) Tokenize + Encode all docs
        # TODO optimize for single batch call
        encoded_docs = []
        for doc in docs:
            encoded_sentences = self.tokenizer.batch_encode_plus(doc, add_special_tokens=False)
            # Create "Start of word mask"
            for e in encoded_sentences.encodings:
                e.start_of_word = _get_start_of_word(e.words, e.special_tokens_mask)
            encoded_docs.append(encoded_sentences)

        # 2) Create sequence pairs that utilize full possible length up to max_seq_len
        # TODO make num special tokens more general
        # account for [CLS], [SEP], [SEP]
        max_num_tokens = self.max_seq_len - 3
        for enc_doc in encoded_docs:
            current_chunk = []
            current_length = 0
            i = 0
            while i < len(enc_doc.encodings):
                current_length += len(enc_doc[i].tokens)
                current_chunk.append(enc_doc[i])

                if current_length >= max_num_tokens:
                    # split our list of sequences (=chunk) into two sequences and create a sample out of it
                    # (incl. special tokens and all other masks)
                    sample, num_unused_segments = self._create_sample_bert_style(
                        chunk=current_chunk,
                        random_doc=encoded_docs[random.randint(0, len(encoded_docs)-1)],
                        max_num_tokens=max_num_tokens,
                    )
                    samples.append(sample)
                    i -= num_unused_segments

                    current_chunk = []
                    current_length = 0
                i += 1
        return samples

    def _create_sequence_pairs_no_next_sent(self, docs):
        samples = []
        # flatten into list of sentences
        docs = [sent for doc in docs for sent in doc]
        # Tokenize + Encode masks
        #TODO fill up sequences rather than creating one-sentence-samples to make this more efficient
        encoded_pairs = self.tokenizer.batch_encode_plus(docs,
                                                         max_length=self.max_seq_len,
                                                         truncation=True,
                                                         truncation_strategy="longest_first",
                                                         add_special_tokens=True,
                                                         padding='max_length'
                                                         )

        assert len(encoded_pairs.input_ids) == len(docs)

        # Create "Start of word mask"
        start_of_word = []
        for e in encoded_pairs.encodings:
            start_of_word.append(_get_start_of_word(e.words, e.special_tokens_mask))

        # Create Sample objects
        for idx in range(len(docs)):
            if len(encoded_pairs.input_ids[idx]) == 0:
                logger.warning(
                    f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {docs[idx]}")
                continue

            # We don't populate 'tokenized' here as we skiped the intermediate string token stage abeoce to improve the speed ...
            samples.append(Sample(id=None,
                                  clear_text={"text_a": docs[idx]},
                                  tokenized={"tokens": encoded_pairs.encodings[idx].tokens,
                                             "start_of_word": start_of_word[idx],
                                             "special_tokens_mask": encoded_pairs.encodings[idx].special_tokens_mask,
                                             "offsets": encoded_pairs.encodings[idx].offsets},
                                  features={"input_ids": encoded_pairs.input_ids[idx],
                                            "segment_ids": encoded_pairs.token_type_ids[idx],
                                            "padding_mask": encoded_pairs.attention_mask[idx],
                                            }
                                  ))
        return samples

    def _create_sample_bert_style(self, chunk, random_doc, max_num_tokens, prob_next_sentence=0.5):
        """
        Get one sample from corpus consisting of two sequences. A sequence can consist of more than one sentence.
        With prob. 50% these are two subsequent sequences from one doc. With 50% the second sequence will be a
        random one from another document.

        :param chunk: List of subsequent, tokenized and encoded sentences.
        :type chunk: [Encoding]
        :param random_doc: A random doc where we can sample a random next "sentence" from.
        :type random_doc: [str]
        :param max_num_tokens: Samples are truncated after this many tokens.
        :type max_num_tokens: int
        :return: (Sample, int)
            sample,
            number of unused sentences in chunk
        """
        # edge case: if we have only a single sequence, we split that one in half
        if len(chunk) == 1:
            # Define splitting point
            if int(len(chunk[0].tokens) / 2) >= max_num_tokens:
                boundary = int(max_num_tokens / 2)
            else:
                boundary = int(len(chunk[0].tokens) / 2)

            # Insert special tokens
            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=chunk[0].ids[:boundary],
                                                                        token_ids_1=chunk[0].ids[
                                                                                    boundary:max_num_tokens])

            segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=chunk[0].ids[:boundary],
                                                                              token_ids_1=chunk[0].ids[
                                                                                          boundary:max_num_tokens])

            # TODO make this general for other model types
            start_of_word = [0] + chunk[0].start_of_word[:boundary] + [0] + chunk[0].start_of_word[boundary:max_num_tokens] + [0]
            padding_mask = [1] * len(input_ids)

            assert len(start_of_word) == len(input_ids)
            assert len(padding_mask) == len(input_ids)
            assert len(segment_ids) == len(input_ids)

            sample = Sample(id=None,
                            clear_text= {"text_a": None,
                                        "text_b": None,
                                        "nextsentence_label": True},
                            tokenized= {"start_of_word": start_of_word},
                            features= {"input_ids": input_ids,
                                        "segment_ids": segment_ids,
                                        "padding_mask": padding_mask,
                                        }
            )
            num_unused_segments = 0
            return sample, num_unused_segments
        else:
            # determine how many segments from chunk go into sequence A
            a_end = random.randrange(1, len(chunk))
            sequence_a = chunk[:a_end]
            length_a = sum([len(seq) for seq in sequence_a])

            # Build sequence B
            target_b_length = max_num_tokens - length_a
            # a) .. using actual next sequence
            if (random.random() > prob_next_sentence) and (len(chunk) > 1):
                sequence_b = chunk[a_end:]
                label = True
                num_unused_segments = 0

            # b) ... using random next sequence
            else:
                sequence_b = []
                length_b = 0
                if len(random_doc.encodings) == 1:
                    sequence_b.append(random_doc[0])
                else:
                    # pick random start sentence and then fill up to target length
                    random_start = random.randrange(len(random_doc.encodings)-1)
                    for i in range(random_start, len(random_doc.encodings)):
                        sequence_b.append(random_doc[i])
                        length_b += len(random_doc[i].ids)
                        if length_b >= target_b_length:
                            break

                label = False

                # We didn't use all of the segments in this chunk as we sampled a random sequence => put them back
                num_unused_segments = len(chunk) - a_end

            # Join everything to single sample
            def merge_start_of_word(sequences):
                start_of_word = []
                for s in sequences:
                    start_of_word.extend(s.start_of_word)
                return start_of_word

            start_of_word_a = merge_start_of_word(sequence_a)
            start_of_word_b = merge_start_of_word(sequence_b)

            sequence_a = Encoding.merge(sequence_a)
            sequence_b = Encoding.merge(sequence_b)

            assert len(sequence_a.ids) > 0
            assert len(sequence_b.ids) > 0

            # Insert special tokens
            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=sequence_a.ids,
                                                                        token_ids_1=sequence_b.ids[:target_b_length])

            segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=sequence_a.ids,
                                                                              token_ids_1=sequence_b.ids[:target_b_length])

            # TODO make this general for other model types
            start_of_word = [0] + start_of_word_a + [0] + start_of_word_b[:target_b_length] + [0]
            padding_mask = [1] * len(input_ids)

            if len(input_ids) < self.max_seq_len:
                # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
                pad_idx = self.tokenizer.pad_token_id
                padding = [pad_idx] * (self.max_seq_len - len(input_ids))
                zero_padding = [0] * (self.max_seq_len - len(input_ids))

                input_ids += padding
                padding_mask += zero_padding
                segment_ids += zero_padding
                start_of_word += zero_padding

            assert len(start_of_word) == len(input_ids)
            assert len(padding_mask) == len(input_ids)
            assert len(segment_ids) == len(input_ids)

            sample = Sample(id=None,
                            clear_text={"text_a": None,
                                        "text_b": None,
                                        "nextsentence_label": label},
                            tokenized={"start_of_word": start_of_word},
                            features={"input_ids": input_ids,
                                      "segment_ids": segment_ids,
                                      "padding_mask": padding_mask,
                                      }
                                )

        return sample, num_unused_segments

    def _create_labels(self, sample, vocab_length) -> dict:
        # Mask random words
        input_ids, lm_label_ids = self._mask_random_words(sample.features["input_ids"], vocab_length, token_groups=sample.tokenized["start_of_word"])
        sample.features["lm_label_ids"] = lm_label_ids
        sample.features["input_ids"] = input_ids

        # NSP label
        if self.next_sent_pred:
            # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
            if sample.clear_text["nextsentence_label"]:
                sample.features["nextsentence_label_ids"] = [0]
            else:
                sample.features["nextsentence_label_ids"] = [1]

        assert len(sample.features["input_ids"]) == self.max_seq_len
        assert len(sample.features["padding_mask"]) == self.max_seq_len
        assert len(sample.features["segment_ids"]) == self.max_seq_len
        assert len(sample.features["lm_label_ids"]) == self.max_seq_len

        return sample.features

    def _mask_random_words(self, tokens, vocab_length, token_groups=None, max_predictions_per_seq=20):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        num_masked.
        If token_groups is supplied, whole word masking is applied, so *all* tokens of a word are either masked or not.
        This option was added by the BERT authors later and showed solid improvements compared to the original objective.
        Whole Word Masking means that if we mask all of the wordpieces corresponding to an original word.
        When a word has been split intoWordPieces, the first token does not have any marker and any subsequence
        tokens are prefixed with ##. So whenever we see the ## token, we
        append it to the previous set of word indexes. Note that Whole Word Masking does *not* change the training code
        at all -- we still predict each WordPiece independently, softmaxed over the entire vocabulary.
        This implementation is mainly a copy from the original code by Google, but includes some simplifications.

        :param tokens: tokenized sentence.
        :type tokens: [str]
        :param vocab_length: number of tokens in the vocabulary
        :type vocab_length: int
        :param token_groups: If supplied, only whole groups of tokens get masked. This can be whole words but
        also other types (e.g. spans). Booleans indicate the start of a group.
        :type token_groups: [bool]
        :param max_predictions_per_seq: maximum number of masked tokens
        :type max_predictions_per_seq: int
        :return: (list of int, list of int), masked tokens and related labels for LM prediction
        """
        # 1. Combine tokens to one group (e.g. all subtokens of a word)
        cand_indices = []
        for (i, token) in enumerate(tokens):
            if token == 101 or token == 102 or token == 0:
                continue
            if (token_groups and len(cand_indices) >= 1 and not token_groups[i]):
                cand_indices[-1].append(i)
            else:
                cand_indices.append([i])

        num_to_mask = min(max_predictions_per_seq,
                          max(1, int(round(len(tokens) * self.masked_lm_prob ))))

        random.shuffle(cand_indices)

        output_label = [-1] * len(tokens)
        num_masked = 0
        assert 103 not in tokens #mask token

        # 2. Mask the first groups until we reach the number of tokens we wanted to mask (num_to_mask)
        for index_set in cand_indices:
            if num_masked >= num_to_mask:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if num_masked + len(index_set) > num_to_mask:
                continue

            for index in index_set:
                prob = random.random()
                num_masked += 1
                original_token = tokens[index]
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[index] = 103

                # 10% randomly change token to random token
                # TODO currently custom vocab is not included here
                elif prob < 0.9:
                    tokens[index] = random.randint(0, vocab_length)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label[index] = original_token
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label[index] = 100 # UNK token
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] instead".format(original_token)
                    )

        return tokens, output_label

    def estimate_n_samples(self, filepath, max_docs=500):
        """
        Estimates the number of samples from a given file BEFORE preprocessing.
        Used in StreamingDataSilo to estimate the number of steps before actually processing the data.
        The estimated number of steps will impact some types of Learning Rate Schedules.
        :param filepath: str or Path, file with data used to create samples (e.g. train.txt)
        :param max_docs: int, maximum number of docs to read in & use for our estimate of n_samples
        :return: int, number of samples in the given dataset
        """

        total_lines = sum(1 for line in open(filepath, encoding="utf-8"))
        empty_lines = sum(1 if line == "\n" else 0 for line in open(filepath, encoding="utf-8"))

        if self.next_sent_pred_style == "sentence":
            # one sample = two lines (except last line in doc)
            n_samples = total_lines - (2 * empty_lines)
        elif self.next_sent_pred_style == "bert-style":
            # Original BERT LM training (filling up sequence pairs with sentences until max_seq_len)
            # (This is a very rough heuristic, as we can only estimate the real number of samples AFTER tokenization)
            logging.info(f"Estimating total number of samples ...")
            # read in subset of docs
            if self.max_docs:
                temp = self.max_docs
                self.max_docs = min(max_docs, temp)
                dicts = list(self.file_to_dicts(filepath))
                self.max_docs = temp
            else:
                self.max_docs = max_docs
                dicts = list(self.file_to_dicts(filepath))
                self.max_docs = None
            # count samples
            dicts = [d["doc"] for d in dicts]
            n_samples = len(self._create_sequence_pairs_bert_style(docs=dicts))
            # extrapolate to the whole file
            n_samples = int(n_samples / len(dicts)) * (empty_lines+1)
            logging.info(f"Heuristic estimate of number of samples in {filepath} based on {len(dicts)} docs: {n_samples}")
        else:
            raise NotImplementedError(f"No estimate logic for next_sent_pred_style={self.next_sent_pred_style} implemented")
        return n_samples


#########################################
# QA Processors ####
#########################################

class QAProcessor(Processor):
    """
    This is class inherits from Processor and is the parent to SquadProcessor and NaturalQuestionsProcessor.
    Its main role is to extend the __init__() so that the number of starting, intermediate and end special tokens
    are calculated from the tokenizer and store as attributes. These are used by the child processors in their
    sample_to_features() methods
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_special_tokens_count()

    def initialize_special_tokens_count(self):
        vec = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=["a"],
                                                              token_ids_1=["b"])
        self.sp_toks_start = vec.index("a")
        self.sp_toks_mid = vec.index("b") - self.sp_toks_start - 1
        self.sp_toks_end = len(vec) - vec.index("b") - 1


class SquadProcessor(Processor):
    """ Used to handle the SQuAD dataset"""

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric="squad",
        train_filename=Path("train-v2.0.json"),
        dev_filename=Path("dev-v2.0.json"),
        test_filename=None,
        dev_split=0,
        doc_stride=128,
        max_query_length=64,
        proxies=None,
        max_answers=6,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, can be "squad" or "top_n_accuracy"
        :type metric: str
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param doc_stride: When the document containing the answer is too long it gets split into part, strided by doc_stride
        :type doc_stride: int
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :type max_query_length: int
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        self.target = "classification"
        self.ph_output_type = "per_token_squad"

        assert doc_stride < (max_seq_len - max_query_length), \
            "doc_stride is longer than max_seq_len minus space reserved for query tokens. \nThis means that there will be gaps " \
            "as the passage windows slide, causing the model to skip over parts of the document.\n" \
            "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384)\n " \
            "Or decrease max_query_length"

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_answers = max_answers
        super(SquadProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )
        self._initialize_special_tokens_count()
        if metric and label_list:
            self.add_task("question_answering", metric, label_list)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def dataset_from_dicts(self, dicts, indices, return_baskets=False):
        """
        Convert input dictionaries into a pytorch dataset for Question Answering.
        For this we have an internal representation called "baskets".
        Each basket is a question-document pair.
        Each stage adds or transforms specific information to our baskets.

        @param dicts: dict, input dictionary with SQuAD style information present
        @param indices: list, indices used during multiprocessing so that IDs assigned to our baskets is unique
        @param return_baskets: boolean, weather to return the baskets or not (baskets are needed during inference)
        @param return_problematic: boolean, weather to return the IDs of baskets that created errors during processing
        """
        # Convert to standard format
        pre_baskets = [self.convert_qa_input_dict(x) for x in dicts] # TODO move to input object conversion

        # Tokenize documents and questions
        baskets = tokenize_batch_question_answering(pre_baskets, self.tokenizer, indices)

        # Split documents into smaller passages to fit max_seq_len
        baskets = self._split_docs_into_passages(baskets)

        # Convert answers from string to token space, skip this step for inference
        if not return_baskets:
            baskets = self._convert_answers(baskets)

        # Convert internal representation (nested baskets + samples with mixed types) to pytorch features (arrays of numbers)
        baskets = self._passages_to_pytorch_features(baskets, return_baskets)

        # Convert features into pytorch dataset, this step also removes potential errors during preprocessing
        dataset, tensor_names, baskets = self._create_dataset(baskets)

        # Logging
        if 0 in indices:
            self._log_samples(1, baskets)

        # During inference we need to keep the information contained in baskets.
        if return_baskets:
            return dataset, tensor_names, self.problematic_sample_ids, baskets
        else:
            return dataset, tensor_names, self.problematic_sample_ids


    def file_to_dicts(self, file: str) -> [dict]:
        nested_dicts = read_squad_file(filename=file)
        dicts = [y for x in nested_dicts for y in x["paragraphs"]]
        return dicts

    # TODO use Input Objects instead of this function
    def convert_qa_input_dict(self, infer_dict):
        """ Input dictionaries in QA can either have ["context", "qas"] (internal format) as keys or
        ["text", "questions"] (api format). This function converts the latter into the former. It also converts the
        is_impossible field to answer_type so that NQ and SQuAD dicts have the same format.
        """
        # check again for doc stride vs max_seq_len when. Parameters can be changed for already initialized models (e.g. in haystack)
        assert self.doc_stride < (self.max_seq_len - self.max_query_length), \
            "doc_stride is longer than max_seq_len minus space reserved for query tokens. \nThis means that there will be gaps " \
            "as the passage windows slide, causing the model to skip over parts of the document.\n" \
            "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384)\n " \
            "Or decrease max_query_length"

        try:
            # Check if infer_dict is already in internal json format
            if "context" in infer_dict and "qas" in infer_dict:
                return infer_dict
            # converts dicts from inference mode to data structure used in FARM
            questions = infer_dict["questions"]
            text = infer_dict["text"]
            uid = infer_dict.get("id", None)
            qas = [{"question": q,
                    "id": uid,
                    "answers": [],
                    "answer_type": None} for i, q in enumerate(questions)]
            converted = {"qas": qas,
                         "context": text}
            return converted
        except KeyError:
            raise Exception("Input does not have the expected format")

    def _initialize_special_tokens_count(self):
        vec = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=["a"],
                                                              token_ids_1=["b"])
        self.sp_toks_start = vec.index("a")
        self.sp_toks_mid = vec.index("b") - self.sp_toks_start - 1
        self.sp_toks_end = len(vec) - vec.index("b") - 1

    def _split_docs_into_passages(self, baskets):
        """
        Because of the sequence length limitation of Language Models, the documents need to be divided into smaller
        parts that we call passages.
        """
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        for basket in baskets:
            samples = []
            ########## perform some basic checking
            # TODO, eventually move checking into input validation functions
            # ignore samples with empty context
            if basket.raw["document_text"] == "":
                logger.warning("Ignoring sample with empty context")
                continue
            ########## end checking


            # Calculate the number of tokens that can be reserved for the passage. This is calculated by considering
            # the max_seq_len, the number of tokens in the question and the number of special tokens that will be added
            # when the question and passage are joined (e.g. [CLS] and [SEP])
            passage_len_t = self.max_seq_len - len(basket.raw["question_tokens"][:self.max_query_length]) - n_special_tokens


            # passage_spans is a list of dictionaries where each defines the start and end of each passage
            # on both token and character level
            try:
                passage_spans = get_passage_offsets(basket.raw["document_offsets"],
                                                    self.doc_stride,
                                                    passage_len_t,
                                                    basket.raw["document_text"])
            except Exception as e:
                logger.warning(f"Could not devide document into passages. Document: {basket.raw['document_text'][:200]}\n"
                               f"With error: {e}")
                passage_spans = []

            for passage_span in passage_spans:
                # Unpack each variable in the dictionary. The "_t" and "_c" indicate
                # whether the index is on the token or character level
                passage_start_t = passage_span["passage_start_t"]
                passage_end_t = passage_span["passage_end_t"]
                passage_start_c = passage_span["passage_start_c"]
                passage_end_c = passage_span["passage_end_c"]

                passage_start_of_word = basket.raw["document_start_of_word"][passage_start_t: passage_end_t]
                passage_tokens = basket.raw["document_tokens"][passage_start_t: passage_end_t]
                passage_text = basket.raw["document_text"][passage_start_c: passage_end_c]

                clear_text = {"passage_text": passage_text,
                              "question_text": basket.raw["question_text"],
                              "passage_id": passage_span["passage_id"],
                              }
                tokenized = {"passage_start_t": passage_start_t,
                             "passage_start_c": passage_start_c,
                             "passage_tokens": passage_tokens,
                             "passage_start_of_word": passage_start_of_word,
                             "question_tokens": basket.raw["question_tokens"][:self.max_query_length],
                             "question_offsets": basket.raw["question_offsets"][:self.max_query_length],
                             "question_start_of_word": basket.raw["question_start_of_word"][:self.max_query_length],
                             }
                # The sample ID consists of internal_id and a passage numbering
                sample_id = f"{basket.id_internal}-{passage_span['passage_id']}"
                samples.append(Sample(id=sample_id,
                                      clear_text=clear_text,
                                      tokenized=tokenized))


            basket.samples=samples

        return baskets

    def _convert_answers(self, baskets):
        """
        Converts answers that are pure strings into the token based representation with start and end token offset.
        Can handle multiple answers per question document pair as is common for development/text sets
        """
        for basket in baskets:
            error_in_answer = False
            for num, sample in enumerate(basket.samples):
                # Dealing with potentially multiple answers (e.g. Squad dev set)
                # Initializing a numpy array of shape (max_answers, 2), filled with -1 for missing values
                label_idxs = np.full((self.max_answers, 2), fill_value=-1)

                if error_in_answer or (len(basket.raw["answers"]) == 0):
                    # If there are no answers we set
                    label_idxs[0, :] = 0
                else:
                    # For all other cases we use start and end token indices, that are relative to the passage
                    for i, answer in enumerate(basket.raw["answers"]):
                        # Calculate start and end relative to document
                        answer_len_c = len(answer["text"])
                        answer_start_c = answer["answer_start"]
                        answer_end_c = answer_start_c + answer_len_c - 1

                        # Convert character offsets to token offsets on document level
                        answer_start_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_start_c)
                        answer_end_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_end_c)
                        # TODO remove after testing 'offset_to_token_idx_vecorized()'
                        # answer_start_t2 = offset_to_token_idx(doc_offsets, answer_start_c)
                        # answer_end_t2 = offset_to_token_idx(doc_offsets, answer_end_c)
                        # if (answer_start_t != answer_start_t2) or (answer_end_t != answer_end_t2):
                        #     pass

                        # Adjust token offsets to be relative to the passage
                        answer_start_t -= sample.tokenized["passage_start_t"]
                        answer_end_t -= sample.tokenized["passage_start_t"]

                        # Initialize some basic variables
                        question_len_t = len(sample.tokenized["question_tokens"])
                        passage_len_t = len(sample.tokenized["passage_tokens"])

                        # Check that start and end are contained within this passage
                        if passage_len_t > answer_start_t >= 0 and passage_len_t > answer_end_t > 0:
                            # Then adjust the start and end offsets by adding question and special tokens
                            label_idxs[i][0] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_start_t
                            label_idxs[i][1] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_end_t
                        # If the start or end of the span answer is outside the passage, treat passage as no_answer
                        else:
                            label_idxs[i][0] = 0
                            label_idxs[i][1] = 0

                        ########## answer checking ##############################
                        # TODO, move this checking into input validation functions and delete wrong examples there
                        # Cases where the answer is not within the current passage will be turned into no answers by the featurization fn
                        if answer_start_t < 0 or answer_end_t >= passage_len_t:
                            pass
                        else:
                            doc_text = basket.raw["document_text"]
                            answer_indices = doc_text[answer_start_c: answer_end_c + 1]
                            answer_text = answer["text"]
                            # check if answer string can be found in context
                            if answer_text not in doc_text:
                                logger.warning(f"Answer '{answer['text']}' not contained in context.\n"
                                               f"Example will not be converted for training/evaluation.")
                                error_in_answer = True
                                label_idxs[i][0] = -100  # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break  # Break loop around answers, so the error message is not shown multiple times
                            elif answer_indices.strip() != answer_text.strip():
                                logger.warning(f"Answer using start/end indices is '{answer_indices}' while gold label text is '{answer_text}'.\n"
                                               f"Example will not be converted for training/evaluation.")
                                error_in_answer = True
                                label_idxs[i][0] = -100 # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break # Break loop around answers, so the error message is not shown multiple times
                        ########## end of checking ####################

                sample.tokenized["labels"] = label_idxs

        return baskets

    def _passages_to_pytorch_features(self, baskets, return_baskets):
        """
        Convert internal representation (nested baskets + samples with mixed types) to python features (arrays of numbers).
        We first join question and passages into on large vector.
        Then we add additional vectors for: - #TODO
        """
        for basket in baskets:
            # Add features to samples
            for num, sample in enumerate(basket.samples):
                # Initialize some basic variables
                question_tokens = sample.tokenized["question_tokens"]
                question_start_of_word = sample.tokenized["question_start_of_word"]
                question_len_t = len(question_tokens)
                passage_start_t = sample.tokenized["passage_start_t"]
                passage_tokens = sample.tokenized["passage_tokens"]
                passage_start_of_word = sample.tokenized["passage_start_of_word"]
                passage_len_t = len(passage_tokens)
                sample_id = [int(x) for x in sample.id.split("-")]

                # - Combines question_tokens and passage_tokens into a single vector called input_ids
                # - input_ids also contains special tokens (e.g. CLS or SEP tokens).
                # - It will have length = question_len_t + passage_len_t + n_special_tokens. This may be less than
                #   max_seq_len but never greater since truncation was already performed when the document was chunked into passages
                question_input_ids = sample.tokenized["question_tokens"]
                passage_input_ids = sample.tokenized["passage_tokens"]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=question_input_ids,
                                                                       token_ids_1=passage_input_ids)

                segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=question_input_ids,
                                                                             token_ids_1=passage_input_ids)
                # To make the start index of passage tokens the start manually
                seq_2_start_t = self.sp_toks_start + question_len_t + self.sp_toks_mid

                start_of_word = [0] * self.sp_toks_start + \
                                    question_start_of_word + \
                                    [0] * self.sp_toks_mid + \
                                    passage_start_of_word + \
                                    [0] * self.sp_toks_end

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                padding_mask = [1] * len(input_ids)

                # The passage mask has 1 for tokens that are valid start or ends for QA spans.
                # 0s are assigned to question tokens, mid special tokens, end special tokens and padding
                # Note that start special tokens are assigned 1 since they can be chosen for a no_answer prediction
                span_mask = [1] * self.sp_toks_start
                span_mask += [0] * question_len_t
                span_mask += [0] * self.sp_toks_mid
                span_mask += [1] * passage_len_t
                span_mask += [0] * self.sp_toks_end

                # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
                pad_idx = self.tokenizer.pad_token_id
                padding = [pad_idx] * (self.max_seq_len - len(input_ids))
                zero_padding = [0] * (self.max_seq_len - len(input_ids))

                input_ids += padding
                padding_mask += zero_padding
                segment_ids += zero_padding
                start_of_word += zero_padding
                span_mask += zero_padding

                # TODO possibly remove these checks after input validation is in place
                len_check = len(input_ids) == len(padding_mask) == len(segment_ids) == len(start_of_word) == len(span_mask)
                id_check = len(sample_id) == 3
                label_check = return_baskets or len(sample.tokenized.get("labels",[])) == self.max_answers
                label_check2 = return_baskets or np.all(sample.tokenized["labels"] > -99) # labels are set to -100 when answer cannot be found
                if len_check and id_check and label_check and label_check2:
                    # - The first of the labels will be used in train, and the full array will be used in eval.
                    # - start_of_word and spec_tok_mask are not actually needed by model.forward() but are needed for
                    #   model.formatted_preds() during inference for creating answer strings
                    # - passage_start_t is index of passage's first token relative to document
                    feature_dict = {"input_ids": input_ids,
                                    "padding_mask": padding_mask,
                                    "segment_ids": segment_ids,
                                    "passage_start_t": passage_start_t,
                                    "start_of_word": start_of_word,
                                    "labels": sample.tokenized.get("labels",[]),
                                    "id": sample_id,
                                    "seq_2_start_t": seq_2_start_t,
                                    "span_mask": span_mask}
                    sample.features = [feature_dict] # other processor's features can be lists
                else:
                    self.problematic_sample_ids.add(sample.id)
                    sample.features = None
        return baskets

    def _create_dataset(self, baskets):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat = []
        basket_to_remove = []
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, baskets

    def _log_samples(self, n_samples, baskets):
        logger.info("*** Show {} random examples ***".format(n_samples))
        for i in range(n_samples):
            random_basket = random.choice(baskets)
            random_sample = random.choice(random_basket.samples)
            logger.info(random_sample)


class NaturalQuestionsProcessor(QAProcessor):
    """ Used to handle the Natural Question QA dataset"""

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename=Path("train-v2.0.json"),
        dev_filename=Path("dev-v2.0.json"),
        test_filename=None,
        dev_split=0,
        doc_stride=128,
        max_query_length=64,
        proxies=None,
        keep_no_answer=0.02,
        downsample_context_size=None,
        inference=False,
        max_answers=6,
        **kwargs):
        """
        Deals with all the preprocessing steps needed for Natural Questions. Follows Alberti 2019 et al. (https://arxiv.org/abs/1901.08634)
        in merging multiple disjoint short answers into the one longer label span and also by downsampling
        samples of no_answer during training

        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing the test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param doc_stride: When the document containing the answer is too long it gets split into parts, strided by doc_stride
        :type doc_stride: int
        :param max_query_length: Maximum length of the question (in number of subword tokens)
        :type max_query_length: int
        :param keep_no_answer: The probability that a sample with an no_answer label is kept
                                    (0.0 < keep_no_answer <= 1.0). Only works if inference is False
        :type keep_no_answer: float
        :param downsample_context_size: Downsampling before any data conversion by taking a short text window of size
                                        downsample_context_size around the long answer span. To disable set to None
        :type downsample_context_size: int
        :param inference: Whether we are currently using the Processsor for model inference. If True, the
                          keep_no_answer will be overridden and set to 1
        :type inference: bool
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        self.target = "classification"
        self.ph_output_type = "per_token_squad"

        # These are classification labels from Natural Questions. Note that in this implementation, we are merging
        # the "long_answer" and "short_answer" labels into the one "span" label
        self.answer_type_list = ["no_answer", "span", "yes", "no"]

        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.keep_no_answer = keep_no_answer
        self.downsample_context_size = downsample_context_size
        self.inference = inference
        self.max_answers = max_answers

        super(NaturalQuestionsProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies
        )

        # Todo rename metric from squad to maybe QA spans or something like that
        self.add_task("question_answering", "squad", ["start_token", "end_token"])
        self.add_task("text_classification", "f1_macro", self.answer_type_list, label_name="answer_type")

    def file_to_dicts(self, file: str) -> [dict]:
        dicts = read_jsonl(file, proxies=self.proxies)
        return dicts

    def _dict_to_samples(self, dictionary: dict, all_dicts=None) -> [Sample]:
        """
            This method will split question-document pairs from the SampleBasket into question-passage pairs which will
        each form one sample. The "t" and "c" in variables stand for token and character respectively. This uses many
        methods that the SquadProcessor calls but note that the SquadProcessor overwrites Processor._dicts_to_baskets()
        while the NaturalQuestionsProcessor does not. This was done in Squad to avoid retokenizing documents that are
        paired with multiple questions. This is not necessary for Natural Questions since there is generally a 1 to 1
        mapping from document to question. Input dictionaries can have either ["context", "qas"] (internal format) as
        keys or ["text", "questions"] (api format). Both are supported.
        """
        # Turns NQ dictionaries into a SQuAD style dictionaries
        if self._is_nq_dict(dictionary):
            dictionary = self._prepare_dict(dictionary=dictionary)

        dictionary_tokenized = self._apply_tokenization(dictionary, self.tokenizer, self.answer_type_list)[0]
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        samples = create_samples_qa_Natural_Question(dictionary_tokenized,
                                    self.max_query_length,
                                    self.max_seq_len,
                                    self.doc_stride,
                                    n_special_tokens)
        # Downsample the number of samples with an no_answer label. This fn will always return at least one sample
        # so that we don't end up with a basket with 0 samples
        if not self.inference:
            samples = self._downsample(samples, self.keep_no_answer)
        return samples

    @staticmethod
    def _is_nq_dict(dictionary):
        if set(dictionary.keys()) == {'document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'}:
            return True
        return False

    def _downsample(self, samples, keep_prob):
        # Downsamples samples with a no_answer label (since there is an overrepresentation of these in NQ)
        # This method will always return at least one sample. This is done so that we don't end up with SampleBaskets
        # with 0 samples
        ret = []
        for s in samples:
            if self._check_no_answer_sample(s):
                if random_float() > 1 - keep_prob:
                    ret.append(s)
            else:
                ret.append(s)
        if len(ret) == 0:
            ret = [random.choice(samples)]
        return ret

    def _downsample_unprocessed(self, dictionary):
        doc_text = dictionary["document_text"]
        doc_tokens = doc_text.split(" ")
        annotations = dictionary.get("annotations",[])
        # for simplicity we only downsample wiki pages with one long answer annotation
        if len(annotations) == 1:
            annotation = annotations[0]
            # There seem to be cases where there is no answer but an annotation is given as a (-1, -1) long answer
            if self._check_no_answer(annotation):
                dictionary["document_text"] = " ".join(doc_tokens[:self.max_seq_len+randint(1,self.downsample_context_size)])
            else:
                # finding earliest start and latest end labels
                long_answer_start = annotation['long_answer']['start_token']
                long_answer_end = annotation['long_answer']['end_token']
                short_answer_start = 1e10
                short_answer_end = -1
                for s in annotation["short_answers"]:
                    if s["start_token"] < short_answer_start:
                        short_answer_start = s["start_token"]
                    if s["end_token"] > short_answer_end:
                        short_answer_end = s["end_token"]

                start_threshold = min(long_answer_start,short_answer_start) - randint(1,self.downsample_context_size)
                start_threshold = max(0, start_threshold)
                end_threshold = max(long_answer_end,short_answer_end) + randint(1,self.downsample_context_size)

                # taking subset of doc text and shift labels
                sub_document_text = " ".join(
                    doc_tokens[start_threshold:end_threshold]
                )
                dictionary["document_text"] = sub_document_text
                # change of offsets happens in place (of dictionary)
                annotation['long_answer']['start_token'] -= start_threshold
                annotation['long_answer']['end_token'] -= start_threshold
                for s in annotation["short_answers"]:
                    s["start_token"] -= start_threshold
                    s["end_token"] -= start_threshold

        return dictionary


    def _prepare_dict(self, dictionary):
        """ Casts a Natural Questions dictionary that is loaded from a jsonl file into SQuAD format so that
        the same featurization functions can be called for both tasks. Each annotation can be one of four answer types,
        ["yes", "no", "span", "no_answer"]"""

        if self.downsample_context_size is not None:
            dictionary = self._downsample_unprocessed(dictionary)

        converted_answers = []
        doc_text = dictionary["document_text"]
        _, tok_to_ch = split_with_metadata(doc_text)
        for annotation in dictionary["annotations"]:
            # There seem to be cases where there is no answer but an annotation is given as a (-1, -1) long answer
            if self._check_no_answer(annotation):
                continue
            sa_text, sa_start_c = self._unify_short_answers(annotation["short_answers"], doc_text, tok_to_ch)
            la_text, la_start_c = self._retrieve_long_answer(annotation["long_answer"]["start_token"],
                                                             annotation["long_answer"]["end_token"],
                                                             tok_to_ch,
                                                             doc_text)
            # Picks the span to be considered as annotation by choosing between short answer, long answer and no_answer
            text, start_c = self._choose_span(sa_text, sa_start_c, la_text, la_start_c)
            converted_answers.append({"text": text,
                                      "answer_start": start_c})
        if len(converted_answers) == 0:
            answer_type = "no_answer"
        else:
            answer_type = dictionary["annotations"][0]["yes_no_answer"].lower()
            if answer_type == "none":
                answer_type = "span"
        # TODO: answer_type should be in answers since in NQ, each annotator can give either a span, no_answer, yes or no
        converted = {"id": dictionary["example_id"],
                     "context": doc_text,
                     "qas": [{"question": dictionary["question_text"],
                              "id": dictionary["example_id"],
                              "answers": converted_answers,
                              "answer_type": answer_type}]}
        return converted

    @staticmethod
    def _check_no_answer(annotation):
        if annotation["long_answer"]["start_token"] > -1 or annotation["long_answer"]["end_token"] > -1:
            return False
        for sa in annotation["short_answers"]:
            if sa["start_token"] > -1 or sa["end_token"] > -1:
                return False
        else:
            return True

    @staticmethod
    def _check_no_answer_sample(sample):
        sample_tok = sample.tokenized
        if len(sample_tok["answers"]) == 0:
            return True
        first_answer = sample_tok["answers"][0]
        if first_answer["start_t"] < sample_tok["passage_start_t"]:
            return True
        if first_answer["end_t"] > sample_tok["passage_start_t"] + len(sample_tok["passage_tokens"]):
            return True
        if first_answer["answer_type"] == "no_answer":
            return True
        else:
            return False

    def _retrieve_long_answer(self, start_t, end_t, tok_to_ch, doc_text):
        """ Retrieves the string long answer and also its starting character index"""
        start_c, end_c = self._convert_tok_to_ch(start_t, end_t, tok_to_ch, doc_text)
        text = doc_text[start_c: end_c]
        return text, start_c

    @staticmethod
    def _choose_span(sa_text, sa_start_c, la_text, la_start_c):
        if sa_text:
            return sa_text, sa_start_c
        elif la_text:
            return la_text, la_start_c
        else:
            return "", -1

    def _unify_short_answers(self, short_answers, doc_text, tok_to_ch):
        """ In cases where an NQ sample has multiple disjoint short answers, this fn generates the single shortest
        span that contains all the answers"""
        if not short_answers:
            return "", -1
        short_answer_idxs = []
        # TODO write comment explaining this
        for short_answer in short_answers:
            short_answer_idxs.append(short_answer["start_token"])
            short_answer_idxs.append(short_answer["end_token"])
        answer_start_t = min(short_answer_idxs)
        answer_end_t = max(short_answer_idxs)
        answer_start_c, answer_end_c = self._convert_tok_to_ch(answer_start_t, answer_end_t, tok_to_ch, doc_text)
        answer_text = doc_text[answer_start_c: answer_end_c]
        assert answer_text == " ".join(doc_text.split()[answer_start_t: answer_end_t])
        return answer_text, answer_start_c

    @staticmethod
    def _convert_tok_to_ch(start_t, end_t, tok_to_ch, doc_text):
        n_tokens = len(tok_to_ch)
        if start_t == -1 and end_t == -1:
            return -1, -1
        start_c = tok_to_ch[start_t]
        # when the end of the answer span is the end of the text
        if end_t == n_tokens:
            end_c = len(doc_text)
        else:
            next_word_start_c = tok_to_ch[end_t]
            span = doc_text[:next_word_start_c].strip()
            end_c = len(span)
        return start_c, end_c

    def _sample_to_features(self, sample: Sample) -> dict:
        self._check_valid_answer(sample)
        features = sample_to_features_qa_Natural_Questions(sample=sample,
                                         tokenizer=self.tokenizer,
                                         max_seq_len=self.max_seq_len,
                                         sp_toks_start=self.sp_toks_start,
                                         sp_toks_mid=self.sp_toks_mid,
                                         sp_toks_end=self.sp_toks_end,
                                         answer_type_list=self.answer_type_list,
                                         max_answers=self.max_answers)
        return features

    def _check_valid_answer(self, sample):
        passage_text = sample.clear_text["passage_text"]
        for answer in sample.clear_text["answers"]:
            len_passage = len(passage_text)
            start = answer["start_c"]
            end = answer["end_c"]
            # Cases where the answer is not within the current passage will be turned into no answers by the featurization fn
            if start < 0 or end >= len_passage:
                continue
            answer_indices = passage_text[start: end + 1]
            answer_text = answer["text"]
            if answer_indices != answer_text:
                raise ValueError(
                    f"""Answer using start/end indices is '{answer_indices}' while gold label text is '{answer_text}'""")

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """
            This method will split the question-document pair from the dictionary into question-passage pairs which will
        each form one sample. The "t" and "c" in variables stand for token and character respectively.
        Input dictionaries can have either ["context", "qas"] (internal format) as keys or ["text", "questions"]
        (api format). Both are supported.
        """
        if self._is_nq_dict(dictionary):
            dictionary = self._prepare_dict(dictionary=dictionary)
        basket_id_internal = kwargs["basket_id_internal"]

        dictionary_tokenized = self._apply_tokenization(dictionary, self.tokenizer, self.answer_type_list)[0]
        n_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        samples = create_samples_qa_Natural_Question(dictionary_tokenized,
                                    self.max_query_length,
                                    self.max_seq_len,
                                    self.doc_stride,
                                    n_special_tokens)
        # Downsample the number of samples with an no_answer label. This fn will always return at least one sample
        # so that we don't end up with a basket with 0 samples.
        if not self.inference:
            samples = self._downsample(samples, self.keep_no_answer)

        # Get features for each sample
        for num, sample in enumerate(samples):
            sample.id = f"{basket_id_internal}-{num}"
            features = self._sample_to_features(sample)
            sample.features = features

        return samples

    def _apply_tokenization(self, dictionary, tokenizer, answer_types_list=[]):
        raw_baskets = []
        dictionary = convert_qa_input_dict(dictionary)
        dictionary["qas"] = self._is_impossible_to_answer_type(dictionary["qas"])
        document_text = dictionary["context"]

        document_tokenized = tokenize_with_metadata(document_text, tokenizer)
        document_start_of_word = [int(x) for x in document_tokenized["start_of_word"]]
        questions = dictionary["qas"]
        for question in questions:
            answers = []
            # For training and dev with labelled examples
            try:
                external_id = question["id"]
                question_text = question["question"]
                for answer in question["answers"]:
                    if 'answer_type' in answer.keys() and answer['answer_type'] in answer_types_list:
                        answer_type = answer['answer_type']
                    else:
                        if answer["text"] == "":
                            answer_type = "no_answer"
                        else:
                            answer_type = "span"
                    a = {"text": answer["text"],
                         "offset": answer["answer_start"],
                         "answer_type": answer_type}
                    answers.append(a)
            # For inference where samples are read in as dicts without an id or answers
            except TypeError:
                external_id = try_get(ID_NAMES, dictionary)
                question_text = question

            question_tokenized = tokenize_with_metadata(question_text, tokenizer)
            question_start_of_word = [int(x) for x in question_tokenized["start_of_word"]]

            # During inference, there is no_answer type. Also, question might be a str instead of a dict
            if type(question) == str:
                answer_type = None
            elif type(question) == dict:
                answer_type = question.get("answer_type", None)
            else:
                raise Exception("Question was neither in str nor dict format")

            raw = {"document_text": document_text,
                   "document_tokens": document_tokenized["tokens"],
                   "document_offsets": document_tokenized["offsets"],
                   "document_start_of_word": document_start_of_word,
                   "question_text": question_text,
                   "question_tokens": question_tokenized["tokens"],
                   "question_offsets": question_tokenized["offsets"],
                   "question_start_of_word": question_start_of_word,
                   "answers": answers,
                   "answer_type": answer_type,
                   "external_id": external_id}
            raw_baskets.append(raw)
        return raw_baskets

    def _is_impossible_to_answer_type(self, qas):
        """ Converts questions from having an is_impossible field to having an answer_type field"""
        new_qas = []
        for q in qas:
            answer_type = "span"
            if "is_impossible" in q:
                if q["is_impossible"] == True:
                    answer_type = "no_answer"
                del q["is_impossible"]
                q["answer_type"] = answer_type
            new_qas.append(q)
        return new_qas


class RegressionProcessor(Processor):
    """
    Used to handle a regression dataset in tab separated text + label
    """
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        label_name="regression_label",
        scaler_mean=None,
        scaler_scale=None,
        proxies=None,
        text_column_name="text",
        **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a
                 numerical value. For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param label_name: name for the internal label variable in FARM (only needed to adjust in rare cases)
        :type label_name: str
        :param scaler_mean: Value to substract from the label for normalization
        :type scaler_mean: float
        :param scaler_scale: Value to divide the label by for normalization
        :type scaler_scale: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows

        super(RegressionProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            proxies=proxies
        )

        # Note that label_list is being hijacked to store the scaling mean and scale
        self.add_task(name="regression",
                      metric="mse",
                      label_list=[scaler_mean, scaler_scale],
                      label_column_name=label_column_name,
                      task_type="regression",
                      label_name=label_name,
                      text_column_name=text_column_name)

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {}
        for task in self.tasks.values():
            column_mapping[task["label_column_name"]] = task["label_name"]
            column_mapping[task["text_column_name"]] = "text"
        dicts = read_tsv(
            rename_columns=column_mapping,
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            proxies=self.proxies
        )

        # collect all labels and compute scaling stats
        train_labels = []
        if self.train_filename in str(file):
            for d in dicts:
                train_labels.append(float(d[self.tasks["regression"]["label_name"]]))
            scaler = StandardScaler()
            scaler.fit(np.reshape(train_labels, (-1, 1)))
            # add to label list in regression task
            self.tasks["regression"]["label_list"] = [scaler.mean_.item(), scaler.scale_.item()]

        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        if len(tokenized["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(seq_a=tokenized[seq_name], seq_b=None,
                                                           tokenizer=self.tokenizer,
                                                           max_seq_len=self.max_seq_len)
        # Samples don't have labels during Inference mode
        label_key = self.tasks["regression"]["label_name"]
        if label_key in dictionary:
            label = float(dictionary[label_key])
            scaled_label = (label - self.tasks["regression"]["label_list"][0]) / self.tasks["regression"]["label_list"][1]
            dictionary[label_key] = scaled_label
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer
        )
        return features

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        """This method is used so that we need to tokenize only once when using a fast tokenizer."""
        text = dictionary["text"]
        inputs = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        # Get tokens as text with metadata
        tokens = []
        offsets = []
        start_of_word = []
        previous_token_end = -1
        for token_id, is_special_token, offset in zip(inputs["input_ids"],
                                                      inputs["special_tokens_mask"],
                                                      inputs["offset_mapping"]):
            if not is_special_token:
                tokens.append(self.tokenizer.convert_ids_to_tokens(token_id))
                offsets.append(offset[0])
                start_of_word.append(True if offset[0] != previous_token_end else False)
                previous_token_end = offset[1]

        token_dict = {"tokens": tokens,
                      "offsets": offsets,
                      "start_of_word": start_of_word}

        if len(token_dict["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}")
            return []

        # Build feature dict
        input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)
        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
        if "XLNetTokenizer" in self.tokenizer.__class__.__name__:
            pad_on_left = True
            segment_ids = pad(segment_ids, self.max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(segment_ids, self.max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(input_ids, self.max_seq_len, self.tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(padding_mask, self.max_seq_len, 0, pad_on_left=pad_on_left)

        assert len(input_ids) == self.max_seq_len
        assert len(padding_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        feat_dict = {"input_ids": input_ids,
                     "padding_mask": padding_mask,
                     "segment_ids": segment_ids}

        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            try:
                label_name = task["label_name"]
                label_raw = dictionary[label_name]
                if task["task_type"] == "regression":
                    label_ids = [float(label_raw)]
                else:
                    raise ValueError(task["task_type"])
            except KeyError:
                # For inference mode we don't expect labels
                label_ids = None
            if label_ids is not None:
                feat_dict[task["label_tensor_name"]] = label_ids

        return [Sample(id=None, clear_text=dictionary, tokenized=token_dict, features=[feat_dict])]


class TextSimilarityProcessor(Processor):
    """
    Used to handle the text DPR datasets that come in json format, example: nq-train.json, nq-dev.json, trivia-train.json, trivia-dev.json
    Datasets can be downloaded from the official DPR github repository (https://github.com/facebookresearch/DPR)

    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {"dataset": str,
    "question": str,
    "answers": list of str
    "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    Example of 1 sample in DPR data json:
    {
    "dataset": "nq_dev_psgs_w100",
    "question": "who sings does he love me with reba",
    "answers": ["Linda Davis"],
    "positive_ctxs": [
    {
    "title": "Does He Love You",
    "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
    "score": 1000,
    "title_score": 1,
    "passage_id": "11828866"
    },
    {
    "title": "Does He Love You",
    "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
    "score": 13.394315,
    "title_score": 0,
    "passage_id": "11828866"
    }, .... ]
    "negative_ctxs": [
    {
    "title": "Cormac McCarthy",
    "text": "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from \"The New York Times\", \"McCarthy doesn't drink anymore \u2013 he quit 16 years ago in El Paso, with one of his young",
    "score": 0,
    "title_score": 0,
    "passage_id": "2145653"
    },
    {
    "title": "Pragmatic Sanction of 1549",
    "text": "one heir, Charles effectively united the Netherlands as one entity. After Charles' abdication in 1555, the Seventeen Provinces passed to his son, Philip II of Spain. The Pragmatic Sanction is said to be one example of the Habsburg contest with particularism that contributed to the Dutch Revolt. Each of the provinces had its own laws, customs and political practices. The new policy, imposed from the outside, angered many inhabitants, who viewed their provinces as distinct entities. It and other monarchical acts, such as the creation of bishoprics and promulgation of laws against heresy, stoked resentments, which fired the eruption of",
    "score": 0,
    "title_score": 0,
    "passage_id": "2271902"
    }, ..... ]
    "hard_negative_ctxs": [
    {
    "title": "Why Don't You Love Me (Beyonce\u0301 song)",
    "text": "song. According to the lyrics of \"Why Don't You Love Me\", Knowles impersonates a woman who questions her love interest about the reason for which he does not value her fabulousness, convincing him she's the best thing for him as she sings: \"Why don't you love me... when I make me so damn easy to love?... I got beauty... I got class... I got style and I got ass...\". The singer further tells her love interest that the decision not to choose her is \"entirely foolish\". Originally released as a pre-order bonus track on the deluxe edition of \"I Am...",
    "score": 14.678405,
    "title_score": 0,
    "passage_id": "14525568"
    },
    {
    "title": "Does He Love You",
    "text": "singing the second chorus. Reba stays behind the wall the whole time, while Linda is in front of her. It then briefly goes back to the dressing room, where Reba continues to smash her lover's picture. The next scene shows Reba approaching Linda's house in the pouring rain at night, while Linda stands on her porch as they sing the bridge. The scene then shifts to the next day, where Reba watches from afar as Linda and the man are seen on a speedboat, where he hugs her, implying that Linda is who he truly loves. Reba finally smiles at",
    "score": 14.385411,
    "title_score": 0,
    "passage_id": "11828871"
    }, ...]
    """
    def __init__(
        self,
        tokenizer,
        passage_tokenizer,
        max_seq_len_query,
        max_seq_len_passage,
        data_dir="",
        metric=None,
        train_filename="train.json",
        dev_filename=None,
        test_filename="test.json",
        dev_split=0.1,
        proxies=None,
        max_samples=None,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
        shuffle_negatives=True,
        shuffle_positives=False,
        label_list=None,
        **kwargs
    ):
        """
        :param tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a passage (str) into tokens.
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :type max_seq_len_query: int
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :type max_seq_len_passage: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param max_samples: maximum number of samples to use
        :type max_samples: int
        :param embed_title: Whether to embed title in passages during tensorization (bool),
        :param num_hard_negatives: maximum number to hard negative context passages in a sample
        :param num_positives: maximum number to positive context passages in a sample
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the num_hard_negative number of passages
        :type shuffle_negatives: bool
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the num_positive number of passages
        :type shuffle_positives: bool
        :param label_list: list of labels to predict. Usually ["hard_negative", "positive"]
        :type label_list: list[str]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        #TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.embed_title = embed_title
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage

        super(TextSimilarityProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len_query,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(name="text_similarity",
                          metric=metric,
                          label_list=label_list,
                          label_name="label",
                          task_type="text_similarity")
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Overwriting method from parent class to **always** load the TextSimilarityProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an TextSimilarityProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        # init tokenizer
        tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["tokenizer"])
        passage_tokenizer = Tokenizer.load(load_dir, tokenizer_class=config["passage_tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]
        del config["passage_tokenizer"]

        processor = cls.load(tokenizer=tokenizer, passage_tokenizer=passage_tokenizer, processor_name="TextSimilarityProcessor", **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__
        config["passage_tokenizer"] = self.passage_tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))
        self.passage_tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def file_to_dicts(self, file: str) -> [dict]:
        """
        Converts a Dense Passage Retrieval (DPR) data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format

        Returns:
        list of dictionaries: List[dict]
        each dictionary:
        {"query": str,
        "passages": [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
        {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
        ...]}
        """
        dicts = read_dpr_json(file, max_samples=self.max_samples)
        return dicts

    def _normalize_question(self, question: str) -> str:
        """
        Removes '?' from queries/questions

        :param question: string representing the question

        Returns:
            Question without the '?'
        """
        if question[-1] == '?':
            question = question[:-1]
        return question

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        Creates one sample from one dict consisting of the query, positive passages and hard negative passages
        :param dictionary:  {"query": str,
                            "passages": List[
                                            {'title': str,
                                            'text': str,
                                            'label': 'hard_negative',
                                            'external_id': str},
                                            {'title': str,
                                            'text': str,
                                            'label': 'positive',
                                            'external_id': str},
                                            ....
                                            ]
                            }

        Returns:
                sample: instance of Sample
        """


        clear_text = {}
        tokenized = {}
        features = {}
        # extract query, positive context passages and titles, hard-negative passages and titles
        if "query" in dictionary.keys():
            query = self._normalize_question(dictionary["query"])

            # featurize the query
            query_inputs = self.query_tokenizer.encode_plus(
                text=query,
                max_length=self.max_seq_len_query,
                add_special_tokens=True,
                truncation=True,
                truncation_strategy='longest_first',
                padding="max_length",
                return_token_type_ids=True,
            )

            query_input_ids, query_segment_ids, query_padding_mask = query_inputs["input_ids"], query_inputs[
                "token_type_ids"], query_inputs["attention_mask"]

            # tokenize query
            tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_input_ids)

            if len(tokenized_query) == 0:
                logger.warning(
                    f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize")
                return None

            clear_text["query_text"] = query
            tokenized["query_tokens"] = tokenized_query
            features["query_input_ids"] = query_input_ids
            features["query_segment_ids"] = query_segment_ids
            features["query_attention_mask"] = query_padding_mask

        if "passages" in dictionary.keys():
            positive_context = list(filter(lambda x: x["label"] == "positive", dictionary["passages"]))
            if self.shuffle_positives:
                random.shuffle(positive_context)
            positive_context = positive_context[:self.num_positives]
            hard_negative_context = list(filter(lambda x: x["label"] == "hard_negative", dictionary["passages"]))
            if self.shuffle_negatives:
                random.shuffle(hard_negative_context)
            hard_negative_context = hard_negative_context[:self.num_hard_negatives]

            positive_ctx_titles = [passage.get("title", None) for passage in positive_context]
            positive_ctx_texts = [passage["text"] for passage in positive_context]
            hard_negative_ctx_titles = [passage.get("title", None) for passage in hard_negative_context]
            hard_negative_ctx_texts = [passage["text"] for passage in hard_negative_context]

            # all context passages and labels: 1 for positive context and 0 for hard-negative context
            ctx_label = [1]*self.num_positives + [0]*self.num_hard_negatives #(self.num_positives if self.num_positives < len(positive_context) else len(positive_context)) + \
            # +(self.num_hard_negatives if self.num_hard_negatives < len(hard_negative_context) else len(hard_negative_context))

            # featurize context passages
            if self.embed_title:
                # concatenate title with positive context passages + negative context passages
                def _combine_title_context(titles, texts):
                    res = []
                    for title, ctx in zip(titles, texts):
                        if title is None:
                            title = ""
                            logger.warning(
                                f"Couldn't find title although `embed_title` is set to True for DPR. Using title='' now. Related passage text: '{ctx}' ")
                        res.append(tuple((title, ctx)))
                    return res

                all_ctx = _combine_title_context(positive_ctx_titles, positive_ctx_texts) + _combine_title_context(
                    hard_negative_ctx_titles, hard_negative_ctx_texts)
            else:
                all_ctx = positive_ctx_texts + hard_negative_ctx_texts

            # assign empty string tuples if hard_negative passages less than num_hard_negatives
            all_ctx += [('', '')] * ((self.num_positives + self.num_hard_negatives)-len(all_ctx))


            ctx_inputs = self.passage_tokenizer.batch_encode_plus(
                all_ctx,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len_passage,
                return_token_type_ids=True
            )


            ctx_input_ids, ctx_segment_ids_, ctx_padding_mask = ctx_inputs["input_ids"], ctx_inputs["token_type_ids"], \
                                                               ctx_inputs["attention_mask"]
            ctx_segment_ids = list(torch.zeros((len(ctx_segment_ids_), len(ctx_segment_ids_[0]))).numpy())

            # tokenize query and contexts
            tokenized_passage = [self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in ctx_input_ids]

            if len(tokenized_passage) == 0:
                logger.warning(f"The context could not be tokenized, likely because it contains a character that the context tokenizer does not recognize")
                return None

            clear_text["passages"] = positive_context + hard_negative_context
            tokenized["passages_tokens"] = tokenized_passage
            features["passage_input_ids"] = ctx_input_ids
            features["passage_segment_ids"] = ctx_segment_ids
            features["passage_attention_mask"] = ctx_padding_mask
            features["label_ids"] = ctx_label


        sample = Sample(id=None,
                        clear_text=clear_text,
                        tokenized=tokenized,
                        features=features)
        return [sample]

    def _sample_to_features(self, sample) -> dict:
        return [sample.features]

    def _dict_to_samples_and_features(self, dictionary: dict, **kwargs) -> [Sample]:
        samples = self._dict_to_samples(dictionary, **kwargs)
        for sample in samples:
            sample.features = self._sample_to_features(sample)

        return samples



