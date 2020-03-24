from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
from transformers.file_utils import cached_path
from transformers.tokenization_bert import BertTokenizer


# create dictionaries for HF transformers integration
# language model config
PRETRAINED_CONFIG_ARCHIVE_MAP = {"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/language_model_config.json"}
# tokenization
EMBEDDING_VOCAB_FILES_MAP = {}
EMBEDDING_VOCAB_FILES_MAP["vocab_file"] = {"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vocab.txt"}
MAX_MODEL_INPU_SIZES = {"glove-german-uncased": 10000}
PRETRAINED_INIT_CONFIGURATION = {"glove-german-uncased": {"do_lower_case": False}}
#model
EMBEDDING_MODEL_MAP = {"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vectors.txt"}


logger = logging.getLogger(__name__)



def load_embedding_tokenizer(pretrained_model_name_or_path, **kwargs):
    # if the pretrained model points to a file on deepset s3, we need to adjust transformers dictionaries
    if pretrained_model_name_or_path in PRETRAINED_INIT_CONFIGURATION:
        BertTokenizer.pretrained_vocab_files_map["vocab_file"]. \
            update({pretrained_model_name_or_path:EMBEDDING_VOCAB_FILES_MAP["vocab_file"].get(pretrained_model_name_or_path, None)})
        BertTokenizer.max_model_input_sizes. \
            update({pretrained_model_name_or_path:MAX_MODEL_INPU_SIZES.get(pretrained_model_name_or_path,None)})
        BertTokenizer.pretrained_init_configuration. \
            update({pretrained_model_name_or_path:PRETRAINED_INIT_CONFIGURATION.get(pretrained_model_name_or_path,None)})
    ret = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    return ret


def load_model(pretrained_model_name_or_path, **kwargs):
    #### loading CONFIG
    resolved_config_file = load_from_cache(pretrained_model_name_or_path, PRETRAINED_CONFIG_ARCHIVE_MAP, **kwargs)
    config_dict = _dict_from_json_file(resolved_config_file)

    #### loading vocab
    resolved_vocab_file = load_from_cache(pretrained_model_name_or_path, EMBEDDING_VOCAB_FILES_MAP["vocab_file"], **kwargs)

    #### loading model
    resolved_model_file = load_from_cache(pretrained_model_name_or_path, EMBEDDING_MODEL_MAP, **kwargs)

    return config_dict, resolved_vocab_file, resolved_model_file

def load_from_cache(pretrained_model_name_or_path, s3_dict, **kwargs):
    # Load from URL or cache if already cached
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)

    s3_file = s3_dict[pretrained_model_name_or_path]
    try:
        resolved_file = cached_path(
                        s3_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                    )

        if resolved_file is None:
            raise EnvironmentError

    except EnvironmentError:
        if pretrained_model_name_or_path in s3_dict:
            msg = "Couldn't reach server at '{}' to download data.".format(
                s3_file
            )
        else:
            msg = (
                "Model name '{}' was not found in model name list. "
                "We assumed '{}' was a path, a model identifier, or url to a configuration file or "
                "a directory containing such a file but couldn't find any such file at this path or url.".format(
                    pretrained_model_name_or_path, s3_file,
                )
            )
        raise EnvironmentError(msg)

    if resolved_file == s3_file:
        logger.info("loading file {}".format(s3_file))
    else:
        logger.info("loading file {} from cache at {}".format(s3_file, resolved_file))

    return resolved_file

def _dict_from_json_file(json_file: str):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)