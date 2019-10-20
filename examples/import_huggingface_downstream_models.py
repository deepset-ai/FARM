from farm.modeling.prediction_head import SquadHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.tokenization import BertTokenizer
from farm.data_handler.processor import SquadProcessor
from farm.utils import  initialize_device_settings


####################### loads a SQUAD finetuned model
# saves it as a FARM adaptive model
device, n_gpu = initialize_device_settings(use_cuda=True)
model = "hugging_squad"
save_dir = "saved_models/farm-hugging-squad"
lm = Bert.load(model)
ph = SquadHead.load(model)
am = AdaptiveModel(language_model=lm,prediction_heads=[ph],embeds_dropout_prob=0.1,lm_output_types="per_token",device=device)
am.save(save_dir)
# saves the processor associated with it, so you can use it in inference mode
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model, do_lower_case=False)
label_list = ["start_token", "end_token"]
metric = "squad"
processor = SquadProcessor(
    tokenizer=tokenizer,max_seq_len=256,labels=label_list,metric=metric,data_dir="../data/squad20",
)
processor.save(save_dir)