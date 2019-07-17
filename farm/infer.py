import torch

from torch.utils.data.sampler import SequentialSampler

from farm.data_handler.dataloader import NamedDataLoader
from farm.modeling.adaptive_model import AdaptiveModel

from farm.utils import initialize_device_settings
from farm.data_handler.processor import Processor


class Inferencer:
    def __init__(self, load_dir, batch_size=4, gpu=False):
        # Init device and distributed settings
        device, n_gpu = initialize_device_settings(
            use_cuda=gpu, local_rank=-1, fp16=False
        )

        self.processor = Processor.load_from_dir(load_dir)
        self.model = AdaptiveModel.load(load_dir, device)
        self.batch_size = batch_size
        self.device = device
        self.language = self.model.language_model.language
        self.prediction_type = "sequence_classification"
        self.name = "bert"
        self.label_map = {i: label for i, label in enumerate(self.processor.label_list)}

    def run_inference(self, dicts):

        if self.prediction_type != "sequence_classification":
            raise NotImplementedError

        dataset, tensor_names = self.processor.dataset_from_dicts(dicts)
        samples = []
        for dict in dicts:
            samples.extend(self.processor._dict_to_samples(dict))

        data_loader = NamedDataLoader(
            dataset=dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size,
            tensor_names=tensor_names,
        )

        preds_all = []
        for batch in data_loader:
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():
                logits = self.model.forward(**batch)
                preds = self.model.formatted_preds(
                    logits=logits,
                    label_maps=self.processor.label_maps,
                    samples=samples,
                    tokenizer=self.processor.tokenizer,
                    **batch
                )
                preds_all.append(preds)
        return preds_all
