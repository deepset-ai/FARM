
.. image:: https://github.com/deepset-ai/FARM/blob/master/docs/logo_with_name.png?raw=true
    :width: 383
    :height: 116
    :align: left
    :alt: FARM LOGO


(**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels)

What is it?
############
FARM makes cutting edge **Transfer Learning** for NLP simple. 
It is a home for all species of pretrained language models (e.g. BERT) that can be adapted to different down-stream
tasks.
The aim is to make it simple to perform document classification, NER and question answering, for example, using the one language model.
The standardized interfaces for language models and prediction heads allow flexible extension by researchers and easy adaptation for practitioners.
Additional experiment tracking and visualizations support you along the way to adapt a SOTA model to your own NLP problem and showcase it as a PoC.  

Have a look at `this blog post <https://www.digitalminds.io/blog/transfer_learning_entering_a_new_era_in_nlp>`_ for an introduction to Transfer Learning
 or see the `full documentation <https://farm.deepset.ai>`_ for more details about FARM

Core features
##############
- Easy adaptation of pretrained language models (e.g. BERT) to your own use case
   - The Processor class makes it easy to define the data processing needed for your task
- Modular design of language model and prediction heads
   - The language model captures a core language understanding that can be shared across tasks
   - A prediction head uses the output of the language model to perform specific downstream tasks and can be easily tailored to your needs
- Easy experiment tracking & execution
- Simple deployment and visualization to showcase your PoC


Installation
#############

The package is tested with Python 3.7.::

    git clone https://github.com/deepset-ai/FARM.git
    cd FARM
    pip install -r requirements.txt
    pip install .


Basic Usage
############

1. Train a downstream model
****************************
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

    # An AdaptiveModel is the combination of a language model and one or more prediction heads
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

See this `tutorial <https://github.com/deepset-ai/FARM/blob/master/tutorials/1_farm_building_blocks.ipynb>`_ for details.

*Usecases:* Custom datasets, language models, prediction heads ...

Metrics and parameters of your model training get automatically logged via MLflow. We provide a `public MLflow server <https://public-mlflow.deepset.ai/>`_ for testing and learning purposes. Check it out to see your own experiment results!

2. Run Inference (API + UI)
****************************

* Run :code:`docker-compose up`
* Open http://localhost:3000 in your browser

.. image:: https://github.com/deepset-ai/FARM/blob/master/docs/inference-api-screen.png?raw=true
    :alt: FARM Inferennce UI

One docker container exposes a REST API (localhost:5000) and another one runs a simple demo UI (localhost:3000).
You can use both of them individually and mount your own models. Check out the docs for details.


Upcoming features
###################
- More pretrained models XLNet, XLM ...
- SOTA adaptation strategies (Adapter Modules, Discriminative Fine-tuning ...)
- Enabling large scale deployment for production
- Additional Visualizations and statistics to explore and debug your model
