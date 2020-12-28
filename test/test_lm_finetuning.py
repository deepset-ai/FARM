import logging
from pathlib import Path

import numpy as np
import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.experiment import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import BertLMHead, NextSentenceHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.infer import Inferencer

def test_lm_finetuning(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )

    processor = BertStyleLMProcessor(
        data_dir=Path("samples/lm_finetuning"),
        train_filename="train-sample.txt",
        test_filename="test-sample.txt",
        dev_filename=None,
        tokenizer=tokenizer,
        max_seq_len=12,
        next_sent_pred=True
    )
    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)

    language_model = LanguageModel.load(lang_model)
    lm_prediction_head = BertLMHead.load(lang_model)
    next_sentence_head = NextSentenceHead.load(lang_model)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[lm_prediction_head, next_sentence_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token", "per_sequence"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1,
        device=device,
        schedule_opts={'name': 'CosineWarmup', 'warmup_proportion': 0.1})

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        evaluate_every=evaluate_every,
        device=device,
    )

    trainer.train()

    # LM embeddings and weight of decoder in head are shared and should therefore be equal
    assert torch.all(
        torch.eq(model.language_model.model.embeddings.word_embeddings.weight, model.prediction_heads[0].decoder.weight))

    save_dir = Path("testsave/lm_finetuning")
    model.save(save_dir)
    processor.save(save_dir)

    del model
    del processor
    del optimizer
    del data_silo
    del trainer

    basic_texts = [
        {"text": "Farmer's life is great."},
        {"text": "It's nothing for big city kids though."},
    ]
    model = Inferencer.load(save_dir, task_type="embeddings", num_processes=0)
    result = model.extract_vectors(dicts=basic_texts)
    assert result[0]["context"] == ['Farmer', "'", 's', 'life', 'is', 'great', '.']
    assert result[0]["vec"].shape == (768,)
    # TODO check why results vary accross runs with same seed
    assert isinstance(result[0]["vec"][0], np.float32)


def test_lm_finetuning_no_next_sentence(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )

    processor = BertStyleLMProcessor(
        data_dir=Path("samples/lm_finetuning"),
        train_filename="train-sample.txt",
        test_filename="test-sample.txt",
        dev_filename=None,
        tokenizer=tokenizer,
        max_seq_len=12,
        next_sent_pred=False
    )
    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)

    language_model = LanguageModel.load(lang_model)
    lm_prediction_head = BertLMHead.load(lang_model)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[lm_prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1,
        device=device,
        schedule_opts={'name': 'CosineWarmup', 'warmup_proportion': 0.1}
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )

    trainer.train()

    # LM embeddings and weight of decoder in head are shared and should therefore be equal
    assert torch.all(
        torch.eq(model.language_model.model.embeddings.word_embeddings.weight, model.prediction_heads[0].decoder.weight))

    save_dir = Path("testsave/lm_finetuning_no_nsp")
    model.save(save_dir)
    processor.save(save_dir)

    del model
    del processor
    del optimizer
    del data_silo
    del trainer

    basic_texts = [
        {"text": "Farmer's life is great."},
        {"text": "It's nothing for big city kids though."},
    ]
    model = Inferencer.load(save_dir, task_type="embeddings", num_processes=0)
    result = model.extract_vectors(dicts=basic_texts)
    assert result[0]["context"] == ['Farmer', "'", 's', 'life', 'is', 'great', '.']
    assert result[0]["vec"].shape == (768,)
    # TODO check why results vary accross runs with same seed
    assert isinstance(result[0]["vec"][0], np.float32)


def test_lm_finetuning_custom_vocab(caplog):
    caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 1
    batch_size = 1
    evaluate_every = 2
    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )
    tokenizer.add_tokens(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbb", "ccccccccccccccccccccccc"])

    processor = BertStyleLMProcessor(
        data_dir=Path("samples/lm_finetuning"),
        train_filename="train-sample.txt",
        test_filename="test-sample.txt",
        dev_filename=None,
        tokenizer=tokenizer,
        max_seq_len=12,
        next_sent_pred=True
    )
    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)

    language_model = LanguageModel.load(lang_model, n_added_tokens=len(tokenizer.get_added_vocab()))
    lm_prediction_head = BertLMHead.load(lang_model, n_added_tokens=len(tokenizer.get_added_vocab()))
    next_sentence_head = NextSentenceHead.load(lang_model)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[lm_prediction_head, next_sentence_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token", "per_sequence"],
        device=device
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        #optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=1,
        device=device,
        schedule_opts={'name': 'CosineWarmup', 'warmup_proportion': 0.1}
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )

    trainer.train()

    # LM embeddings and weight of decoder in head are shared and should therefore be equal
    assert torch.all(
        torch.eq(model.language_model.model.embeddings.word_embeddings.weight, model.prediction_heads[0].decoder.weight))

    save_dir = Path("testsave/lm_finetuning")
    model.save(save_dir)
    processor.save(save_dir)

    del model
    del processor
    del optimizer
    del data_silo
    del trainer

    basic_texts = [
        {"text": "Farmer's life is great."},
        {"text": "It's nothing for big city kids though."},
    ]
    model = Inferencer.load(save_dir, task_type="embeddings", num_processes=0)
    result = model.extract_vectors(dicts=basic_texts)
    assert result[0]["context"] == ['Farmer', "'", 's', 'life', 'is', 'great', '.']
    assert result[0]["vec"].shape == (768,)
    # TODO check why results vary accross runs with same seed
    assert isinstance(result[0]["vec"][0], np.float32)

if(__name__=="__main__"):
    test_lm_finetuning()