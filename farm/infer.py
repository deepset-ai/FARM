import os
import torch
import logging

from torch.utils.data.sampler import SequentialSampler

from farm.data_handler.dataloader import NamedDataLoader
from farm.modeling.adaptive_model import AdaptiveModel

from farm.utils import initialize_device_settings
from farm.data_handler.processor import Processor, InferenceProcessor
from farm.utils import set_all_seeds


logger = logging.getLogger(__name__)


class Inferencer:
    """
    Loads a saved AdaptiveModel from disk and runs it in inference mode. Can be used for a model with prediction head (down-stream predictions) and without (using LM as embedder).

    Example usage:

    .. code-block:: python

       # down-stream inference
       basic_texts = [
           {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
           {"text": "Martin Müller spielt Handball in Berlin"},
       ]
       model = Inferencer.load(your_model_dir)
       model.run_inference(dicts=basic_texts)
       # LM embeddings
       model.extract_vectors(dicts=basic_texts)

    """

    def __init__(self, model, processor, batch_size=4, gpu=False, name=None):
        """
        Initializes inferencer from an AdaptiveModel and a Processor instance.

        :param model: AdaptiveModel to run in inference mode
        :type model: AdaptiveModel
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :type processor: Processor
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :param name: Name for the current inferencer model, displayed in the REST API
        :type name: string
        :return: An instance of the Inferencer.

        """
        # Init device and distributed settings
        device, n_gpu = initialize_device_settings(
            use_cuda=gpu, local_rank=-1, fp16=False
        )

        self.processor = processor
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.language = self.model.language_model.language
        # TODO adjust for multiple prediction heads
        if len(self.model.prediction_heads) == 1:
            self.prediction_type = self.model.prediction_heads[0].model_type
            #self.label_map = self.processor.label_maps[0]
        elif len(self.model.prediction_heads) == 0:
            self.prediction_type = "embedder"
        # else:
        #     raise NotImplementedError("A model with multiple prediction heads is currently not supported by the Inferencer")
        self.name = name if name != None else f"anonymous-{self.prediction_type}"

        model.connect_heads_with_processor(processor.tasks)
        set_all_seeds(42, n_gpu)

    @classmethod
    def load(cls, load_dir, batch_size=4, gpu=False, embedder_only=False):
        """
        Initializes inferencer from directory with saved model.

        :param load_dir: Directory where the saved model is located.
        :type load_dir: str
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :param embedder_only: If true, a faster processor (InferenceProcessor) is loaded. This should only be used
        for extracting embeddings (no downstream predictions).
        :type embedder_only: bool
        :return: An instance of the Inferencer.
        """

        device, n_gpu = initialize_device_settings(
            use_cuda=gpu, local_rank=-1, fp16=False
        )

        model = AdaptiveModel.load(load_dir, device)
        if embedder_only:
            # model.prediction_heads = []
            processor = InferenceProcessor.load_from_dir(load_dir)
        else:
            processor = Processor.load_from_dir(load_dir)

        name = os.path.basename(load_dir)
        return cls(model, processor, batch_size=batch_size, gpu=gpu, name=name)

    def run_inference(self, dicts):
        """
        Runs down-stream inference using the prediction head.

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicst: [dict]
        :return: dict of predictions

        """
        if self.prediction_type == "embedder":
            raise TypeError(
                "You have called run_inference for a model without any prediction head! "
                "If you want to: "
                "a) ... extract vectors from the language model: call `Inferencer.extract_vectors(...)`"
                f"b) ... run inference on a downstream task: make sure your model path {self.name} contains a saved prediction head"
            )
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
        for i, batch in enumerate(data_loader):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]
            with torch.no_grad():
                logits = self.model.forward(**batch)
                preds = self.model.formatted_preds(
                    logits=logits,
                    samples=batch_samples,
                    tokenizer=self.processor.tokenizer,
                    **batch,
                )
                preds_all += preds

        return preds_all

    def extract_vectors(
        self, dicts, extraction_strategy="cls_token", extraction_layer=-1
    ):
        """
        Converts a text into vector(s) using the language model only (no prediction head involved).

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param extraction_strategy: Strategy to extract vectors. Choices: 'cls_token' (sentence vector),
        'reduce_mean' (sentence vector), reduce_max (sentence vector), 'per_token' (individual token vectors)
        :type extraction_strategy: str
        :param extraction_layer: number of layer from which the embeddings shall be extracted. Default: -1 (very last layer).
        :type: int
        :return: dict of predictions
        """

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
        for i, batch in enumerate(data_loader):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]
            with torch.no_grad():
                preds = self.model.language_model.formatted_preds(
                    extraction_strategy=extraction_strategy,
                    samples=batch_samples,
                    tokenizer=self.processor.tokenizer,
                    extraction_layer=extraction_layer,
                    **batch,
                )
                preds_all += preds

        return preds_all


class FasttextInferencer:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name if name != None else f"anonymous-fasttext"
        self.prediction_type = "embedder"

    @classmethod
    def load(cls, load_dir, batch_size=4, gpu=False, embedder_only=True):
        import fasttext
        if os.path.isfile(load_dir):
            return cls(model=fasttext.load_model(load_dir))
        else:
            logger.error(f"Fasttext model file does not exist at: {load_dir}")

    def extract_vectors(self, dicts, extraction_strategy="reduce_mean"):
        """
        Converts a text into vector(s) using the language model only (no prediction head involved).

        :param dicts: Samples to run inference on provided as a list of dicts. One dict per sample.
        :type dicts: [dict]
        :param extraction_strategy: Strategy to extract vectors. Choices: 'reduce_mean' (mean sentence vector), 'reduce_max' (max per embedding dim), 'CLS'
        :type extraction_strategy: str
        :return: dict of predictions
        """

        preds_all = []
        for d in dicts:
            pred = {}
            pred["context"] = d["text"]
            if extraction_strategy == "reduce_mean":
                pred["vec"] = self.model.get_sentence_vector(d["text"])
            else:
                raise NotImplementedError
            preds_all.append(pred)

        return preds_all
