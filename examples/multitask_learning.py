# fmt: off
import torch
import numpy as np
import pandas as pd
import random
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead, TokenClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

# Generate some dummy data
dummy= [[["hello", "this", "is", "a", "demo"], [1,0,0,0,1], "not sw"], [["hello", "this", "is", "starwars"], [1,0,0,1], "sw"]]
train_data = []
test_data = []
for number in range(100):
  train_data.append(dummy[random.randint(0, 1)])
  test_data.append(dummy[random.randint(0, 1)])

train_df = pd.DataFrame(train_data, columns = ['label', 'trigger', 'label'])
train_df.to_csv("train.csv")

test_df = pd.DataFrame(test_data, columns = ['sentence', 'trigger', 'label'])
test_df.to_csv("test.csv")

from farm.data_handler.processor import Processor
from tokenizers.pre_tokenizers import WhitespaceSplit
from farm.data_handler.samples import (
    Sample,
    SampleBasket,
)
from farm.data_handler.utils import expand_labels

class MTLProcessor(Processor):

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        train_filename,
        test_filename,
        delimiter,
        dev_split=0.0,
        dev_filename=None,
        label_list=None,
        metric=None,
        proxies=None,
        **kwargs
    ):
        self.delimiter = delimiter

        super(MTLProcessor, self).__init__(
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

    def file_to_dicts(self, file: str) -> [dict]:
      dicts = list()
      df = pd.read_csv(file)
      for text, label, tokens in zip(df.sentence.values, df.label.values, df.trigger.values):
        columns = dict()
        text = ast.literal_eval(text)
        tokens = ast.literal_eval(tokens)
        columns["text"] = " ".join(text)
        columns["document_level_task_label"] = label # Key hard-coded
        columns["token_level_task_label"] = list(map(str, tokens)) # Key hard-coded
        dicts.append(columns)
      return dicts

    @staticmethod
    def _get_start_of_word(word_ids):
        words = np.array(word_ids)
        words[words == None] = -1
        start_of_word_single = [0] + list(np.ediff1d(words) > 0)
        start_of_word_single = [int(x) for x in start_of_word_single]
        return start_of_word_single

    # Most of the code is copied from NERProcessor - dataset_from_dicts()
    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, non_initial_token="X"):
      self.baskets = []
      self.pre_tokenizer = WhitespaceSplit()

      texts = [x["text"] for x in dicts]
      words_and_spans = [self.pre_tokenizer.pre_tokenize_str(x) for x in texts]
      words = [[x[0] for x in y] for y in words_and_spans]

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
          is_split_into_words=True,
      )

      for i in range(len(dicts)):
          tokenized = tokenized_batch[i]
          d = dicts[i]
          id_external = self._id_from_dict(d)
          if indices:
              id_internal = indices[i]
          else:
              id_internal = i

          input_ids = tokenized.ids
          segment_ids = tokenized.type_ids
          initial_mask = self._get_start_of_word(tokenized.words)
          assert len(initial_mask) == len(input_ids)

          padding_mask = tokenized.attention_mask

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
                  label_name = task["label_name"]
                  labels_word = d[label_name]
                  label_list = task["label_list"]
                  label_tensor_name = task["label_tensor_name"]

                  if task["task_type"] == "classification":
                      label_ids = [label_list.index(labels_word)]
                  elif task["task_type"] == "ner":
                      labels_token = expand_labels(labels_word, initial_mask, non_initial_token)
                      label_ids = [label_list.index(lt) for lt in labels_token]
              except ValueError:
                  label_ids = None
                  problematic_labels = set(labels_token).difference(set(label_list))
                  print(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                                  f"\nWe found a problem with labels {str(problematic_labels)}")
              except KeyError:
                  label_ids = None
                  # print(f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                  #                 "\nIf your are running in *inference* mode: Don't worry!"
                  #                 "\nIf you are running in *training* mode: Verify you are supplying a proper label list to your processor and check that labels in input data are correct.")
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

      if indices and 0 not in indices:
          pass
      else:
          self._log_samples(1)

      dataset, tensor_names = self._create_dataset()
      ret = [dataset, tensor_names, self.problematic_sample_ids]
      if return_baskets:
          ret.append(self.baskets)
      return tuple(ret)


from sklearn.metrics import f1_score

def custom_f1_score(y_true, y_pred):
  f1_scores = []
  for t, p in zip(y_true, y_pred):
    f1_scores.append(f1_score(t, p, average='macro'))
  return {"f1 macro score" : sum(f1_scores) / len(f1_scores), "total" : len(f1_scores)}


from typing import List
def my_loss_agg(individual_losses: List[torch.Tensor], global_step=None, batch=None):
    loss = torch.sum(individual_losses[0]) + torch.sum(individual_losses[1])
    return loss


DO_LOWER_CASE = False
LANG_MODEL = "bert-base-uncased"
TRAIN_FILE = "/content/train.csv"
# DEV_FILE = "/content/dev.csv"
TEST_FILE = "/content/test.csv"
MAX_SEQ_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
N_EPOCHS = 1
EMBEDS_DROPOUT_PROB = 0.1
EVALUATE_EVERY = 20
DEVICE, N_GPU = initialize_device_settings(use_cuda=True)
set_all_seeds(seed=42)

tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=LANG_MODEL,
    do_lower_case=DO_LOWER_CASE,
    )

TRIGGER_LABELS = ["X", "0", "1"]
LABEL_LIST = ["not sw", "sw"]

processor = MTLProcessor(data_dir = ".", 
                          tokenizer=tokenizer,
                          max_seq_len=128,
                          train_filename=TRAIN_FILE,
                          test_filename=TEST_FILE,
                          delimiter=",",
                          )



from farm.evaluation.metrics import register_metrics
register_metrics('f1_weighted', custom_f1_score)

metric = 'f1_weighted'
processor.add_task(name="document_level_task", label_list=LABEL_LIST, metric="acc", text_column_name="text", label_column_name="label", task_type="classification")
processor.add_task(name="token_level_task", label_list=TRIGGER_LABELS, metric=metric, text_column_name="text", label_column_name="tokens", task_type="ner")


data_silo = DataSilo(processor=processor,
                    batch_size=BATCH_SIZE
                    )

language_model = LanguageModel.load(LANG_MODEL)

document_level_task_head = TextClassificationHead(num_labels=len(LABEL_LIST), task_name="document_level_task")
token_level_task_head = TokenClassificationHead(num_labels=len(TRIGGER_LABELS), task_name="token_level_task")

model = AdaptiveModel(
  language_model=language_model,
  prediction_heads=[document_level_task_head, token_level_task_head],
  embeds_dropout_prob=EMBEDS_DROPOUT_PROB,
  lm_output_types=["per_sequence", "per_token"],
  device=DEVICE,
  loss_aggregation_fn=my_loss_agg)

model, optimizer, lr_schedule = initialize_optimizer(
  model=model,
  device=DEVICE,
  learning_rate=LEARNING_RATE,
  n_batches=len(data_silo.loaders["train"]),
  n_epochs=N_EPOCHS)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  data_silo=data_silo,
                  epochs=N_EPOCHS,
                  n_gpu=N_GPU,
                  lr_schedule=lr_schedule,
                  device=DEVICE,
                  evaluate_every=EVALUATE_EVERY,
                  )

model = trainer.train()
