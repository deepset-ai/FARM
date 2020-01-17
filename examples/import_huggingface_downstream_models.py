from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.tokenization import BertTokenizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.data_handler.processor import SquadProcessor
from farm.utils import  initialize_device_settings


def import_downstream_models():
    ####################### loads a SQUAD finetuned model
    # saves it as a FARM adaptive model
    device, n_gpu = initialize_device_settings(use_cuda=True)
    model = "bert-large-uncased-whole-word-masking-finetuned-squad"
    save_dir = "saved_models/FARM-bert-large-uncased-whole-word-masking-finetuned-squad"
    lm = Bert.load(model)
    ph = QuestionAnsweringHead.load(model)
    am = AdaptiveModel(language_model=lm,prediction_heads=[ph],embeds_dropout_prob=0.1,lm_output_types="per_token",device=device)
    am.save(save_dir)
    # saves the processor associated with it, so you can use it in inference mode
    # TODO load HF's tokenizer_config.json and adjust settings
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model)
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=256,
        label_list=label_list,
        metric=metric,
        data_dir="../data/squad20",
    )
    processor.save(save_dir)


if __name__ == "__main__":
    import_downstream_models()
