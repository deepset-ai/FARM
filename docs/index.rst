.. FARM documentation master file, created by
   sphinx-quickstart on Wed Jul  3 18:08:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FARM!
================================

**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels

What is it?
########
FARM makes cutting edge **Transfer Learning** for NLP simple.
It is a home for all species of pretrained language models (e.g. BERT) that can be adapted to different down-stream
tasks (e.g. NER) by simply switching the prediction head.
The standardized interfaces for language models and prediction heads allow a flexible extension by researchers and an easy adaptation for practitioners.
Additional experiment tracking and visualizations support you along the way to adapt a SOTA model to your own NLP problem and showcase it as a PoC.

Core features
########
- Easy adaptation of pretrained language models (e.g. BERT) to your own use case
- Modular design of language model and prediction heads
- Easy experiment tracking & execution
- Simple deployment and visualization to showcase your PoC

Basic Usage
############

1. Train a model
**********************
FARM offers two modes for model training:

**Option 1: Run experiment(s) from config**::

    from farm.experiment import run_experiment, load_experiments
    experiments = load_experiments("experiments/ner/conll2003_de_config.json")
    run_experiment(experiments[0])

*Use cases:* Training your first model, hyperparameter optimization, evaluating a language model on multiple down-stream tasks.

**Option 2: Stick together your own building blocks**::

    # Basic building blocks for data handling
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=lang_model)
    processor = CONLLProcessor(tokenizer=tokenizer, data_dir="../data/conll03-de", max_seq_len=128)
    ...

    # AdaptiveModel = LanguageModel + PredictionHead(s)
    language_model = Bert.load(lang_model)
    prediction_head = TokenClassificationHead(layer_dims=[768, num_labels])
    model = AdaptiveModel(language_model=language_model, prediction_heads=[prediction_head], ...)
    ...

    # Feed it to a Trainer, which keeps care of growing our model
    trainer = Trainer(optimizer=optimizer, data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=1,
        warmup_linear=warmup_linear,
        evaluate_every=evaluate_every,
        device=device,
    )

    # 7. Let it grow
    model = trainer.train(model)

See this tutorial for details

*Usecases:* Custom datasets, language models, prediction heads ...


2. Serve model via API
**********************

Run :code:`docker run -v save:/home/user/save deepset-ai/farm-inference-api:base-models`

*You can exchange* :code:`save` *with any directory containing trained FARM model(s) .*

3. Run Demo UI
**********************

* Run :code:`docker run -v save:/home/user/save deepset-ai/farm-demo-ui`
* Open http://localhost:80 in your browser

Upcoming features
###################
- More pretrained models XLNet, XLM ...
- SOTA adaptation strategies (Adapter Modules, Discriminative Fine-tuning ...)
- Enabling large scale deployment for production
- Additional Visualizations and statistics to explore and debug your model


API Reference
==================
.. toctree::
   :maxdepth: 4
   :caption: API Reference:

   api/farm


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
