import logging

from farm.data_handler.ner import ConllProcessor
from farm.data_handler.ner import (
    convert_examples_to_features as convert_examples_to_features_ner,
)
from farm.data_handler.ner import (
    convert_examples_to_features_old as convert_examples_to_features_ner_old,
)
from farm.modeling.bert.tokenization import BertTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


processor = ConllProcessor()

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-cased-de-2b-end", do_lower_case=False
)


data_bunch = NewDataBunch.load(
    "data/conll03",
    processor,
    tokenizer,
    32,
    128,
    convert_examples_to_features_ner,
    local_rank=-1,
)

data_bunch_old = NewDataBunch.load(
    "data/conll03",
    processor,
    tokenizer,
    32,
    128,
    convert_examples_to_features_ner_old,
    local_rank=-1,
)


test = data_bunch.loaders["test"]
test_old = data_bunch_old.loaders["test"]

for b in test:
    break

for b_old in test_old:
    break

print(b[0])
print(b_old[0])
