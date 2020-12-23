# TODO enable NQ tests again

# import logging
# from pathlib import Path
# import numpy as np
# import pytest
#
# from farm.data_handler.data_silo import DataSilo
# from farm.data_handler.processor import NaturalQuestionsProcessor
# from farm.modeling.adaptive_model import AdaptiveModel
# from farm.modeling.language_model import LanguageModel
# from farm.modeling.optimization import initialize_optimizer
# from farm.modeling.prediction_head import QuestionAnsweringHead, TextClassificationHead
# from farm.modeling.tokenization import Tokenizer
# from farm.train import Trainer
# from farm.utils import set_all_seeds, initialize_device_settings
# from farm.infer import Inferencer, QAInferencer
#
# @pytest.fixture()
# def distilbert_nq(caplog=None):
#     if caplog:
#         caplog.set_level(logging.CRITICAL)
#
#
#     set_all_seeds(seed=42)
#     device, n_gpu = initialize_device_settings(use_cuda=False)
#     batch_size = 2
#     n_epochs = 1
#     evaluate_every = 4
#     base_LM_model = "distilbert-base-uncased"
#
#     tokenizer = Tokenizer.load(
#         pretrained_model_name_or_path=base_LM_model, do_lower_case=True
#     )
#     processor = NaturalQuestionsProcessor(
#         tokenizer=tokenizer,
#         max_seq_len=20,
#         doc_stride=10,
#         max_query_length=6,
#         train_filename="train_sample.jsonl",
#         dev_filename="dev_sample.jsonl",
#         data_dir=Path("samples/nq")
#     )
#
#     data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)
#     language_model = LanguageModel.load(base_LM_model)
#     qa_head = QuestionAnsweringHead()
#     classification_head = TextClassificationHead(num_labels=len(processor.answer_type_list))
#
#     model = AdaptiveModel(
#         language_model=language_model,
#         prediction_heads=[qa_head, classification_head],
#         embeds_dropout_prob=0.1,
#         lm_output_types=["per_token", "per_sequence"],
#         device=device,
#     )
#
#     model, optimizer, lr_schedule = initialize_optimizer(
#         model=model,
#         learning_rate=2e-5,
#         #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
#         n_batches=len(data_silo.loaders["train"]),
#         n_epochs=n_epochs,
#         device=device
#     )
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         data_silo=data_silo,
#         epochs=n_epochs,
#         n_gpu=n_gpu,
#         lr_schedule=lr_schedule,
#         evaluate_every=evaluate_every,
#         device=device
#     )
#     trainer.train()
#     return model, processor
#
#
# def test_training(distilbert_nq):
#     model, processor = distilbert_nq
#     assert type(model) == AdaptiveModel
#     assert type(processor) == NaturalQuestionsProcessor
#
#
# def test_inference(distilbert_nq, caplog=None):
#     if caplog:
#         caplog.set_level(logging.CRITICAL)
#     model, processor = distilbert_nq
#
#     save_dir = Path("testsave/qa_nq")
#     model.save(save_dir)
#     processor.save(save_dir)
#
#     inferencer = QAInferencer.load(save_dir, batch_size=2, gpu=False, num_processes=0)
#     assert inferencer is not None
#
#     qa_format_1 = [
#         {
#             "questions": ["Who counted the game among the best ever made?"],
#             "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
#         }
#     ]
#     qa_format_2 = [
#         {
#             "qas":["Who counted the game among the best ever made?"],
#             "context": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
#         }
#     ]
#
#     result1 = inferencer.inference_from_dicts(dicts=qa_format_1)
#     result2 = inferencer.inference_from_dicts(dicts=qa_format_2)
#     assert result1 == result2
#
# if __name__ == "__main__":
#     test_training()
#     test_inference()