Examples
================================

You can find exemplary scripts for the major down-stream tasks in :code:`examples/`

Document Classification
##########################
(see :code:`examples/doc_classification.py` for full script)

1.Create a tokenizer::

    tokenizer = Tokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)

2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset::

    processor = GermEval18CoarseProcessor(tokenizer=tokenizer,
                              max_seq_len=128,
                              data_dir="../data/germeval18")

3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets::

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

4. Create an AdaptiveModel
a) which consists of a pretrained language model as a basis::

    language_model = LanguageModel.load(lang_model)

b) and a prediction head on top that is suited for our task => Text classification::

    prediction_head = TextClassificationHead(layer_dims=[768, len(processor.label_list)])

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

5. Create an optimizer::

    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        n_examples=data_silo.n_samples("train"),
        batch_size=batch_size,
        n_epochs=1)

6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time::

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=1,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device)

7. Let it grow::

    model = trainer.train(model)

8. Hooray! You have a model. Store it::

    save_dir = "save/bert-german-GNAD-tutorial"
    model.save(save_dir)
    processor.save(save_dir)

9. Load it & harvest your fruits (Inference)::

    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot ist"},
        {"text": "Martin Müller spielt Fussball"},
    ]
    model = Inferencer(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    print(result)
