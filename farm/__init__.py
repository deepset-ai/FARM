import logging

import torch.multiprocessing as mp

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# reduce verbosity from transformers library
logging.getLogger('transformers.configuration_utils').setLevel(logging.WARNING)

# https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
if "file_descriptor" in mp.get_all_sharing_strategies():
    import resource

    mp.set_sharing_strategy("file_descriptor")

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # seting soft limit to hard limit (=rlimit[1]) minus a small amount to be safe
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1]-512, rlimit[1]))


#### Adding Glove model to transformers
import transformers.tokenization_bert
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"].\
    update({"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vocab.txt"})
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.\
    update({"glove-german-uncased":512})
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION.\
    update({"glove-german-uncased": {"do_lower_case": False}})

# TODO remove, possibly not needed
# import transformers.modeling_bert
# transformers.modeling_bert.BERT_PRETRAINED_MODEL_ARCHIVE_MAP.\
#     update({"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/vectors.txt"})
#
# import transformers.configuration_bert
# transformers.configuration_bert.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.\
#     update({"glove-german-uncased":"https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.4.1/glove-german-uncased/language_model_config.json"})

