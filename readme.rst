
.. image:: https://github.com/deepset-ai/FARM/blob/master/docs/img/logo_with_name.png?raw=true
    :width: 383
    :height: 116
    :align: left
    :alt: FARM LOGO


(**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels)

.. image:: https://img.shields.io/github/release/deepset-ai/farm
	:target: https://github.com/deepset-ai/FARM/releases
	:alt: Release

.. image:: https://img.shields.io/github/license/deepset-ai/farm
	:target: https://github.com/deepset-ai/FARM/blob/master/LICENSE
	:alt: License

.. image:: https://img.shields.io/github/last-commit/deepset-ai/farm
	:target: https://github.com/deepset-ai/FARM/commits/master
	:alt: Last Commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
	:target: https://github.com/ambv/black
	:alt: Last Commit

What is it?
############
FARM makes cutting edge **Transfer Learning** for NLP simple.
It is a home for all species of pretrained language models (e.g. BERT) that can be adapted to different down-stream
tasks.
The aim is to make it simple to perform document classification, NER and question answering, for example, using the one language model.
The standardized interfaces for language models and prediction heads allow flexible extension by researchers and easy adaptation for practitioners.
Additional experiment tracking and visualizations support you along the way to adapt a SOTA model to your own NLP problem and showcase it as a PoC.

Have a look at for an introduction to Transfer Learning
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

Resources
##############
Documentation: `Full documentation <https://farm.deepset.ai>`_
Introduction to transfer learning: `Blog <https://www.digitalminds.io/blog/transfer_learning_entering_a_new_era_in_nlp>`_
Tutorial: `Jupyter notebook <https://github.com/deepset-ai/FARM/blob/master/tutorials/1_farm_building_blocks.ipynb>`_ and `Colab notebook <https://colab.research.google.com/drive/130_7dgVC3VdLBPhiEkGULHmqSlflhmVM>`_

Installation
#############

Recommended (because of active development)::

    git clone https://github.com/deepset-ai/FARM.git
    cd FARM
    pip install -r requirements.txt
    pip install --editable .

If problems occur, please do a git pull. The --editable flag will update changes immediately.

From PyPi::

    pip install farm

Basic Usage
############

1. Train a downstream model
****************************
FARM offers two modes for model training:

**Option 1: Run experiment(s) from config**

.. raw:: html

    <img src="https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/code_snippet_experiment.png">

*Use cases:* Training your first model, hyperparameter optimization, evaluating a language model on multiple down-stream tasks.

**Option 2: Stick together your own building blocks**

.. raw:: html

    <img src="https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/code_snippet_building_blocks.png">

*Usecases:* Custom datasets, language models, prediction heads ...

Metrics and parameters of your model training get automatically logged via MLflow. We provide a `public MLflow server <https://public-mlflow.deepset.ai/>`_ for testing and learning purposes. Check it out to see your own experiment results! Just be aware: We will start deleting all experiments on a regular schedule to ensure decent server performance for everybody!

2. Run Inference (API + UI)
****************************

* Run :code:`docker-compose up`
* Open http://localhost:3000 in your browser

.. image:: https://github.com/deepset-ai/FARM/blob/master/docs/img/inference-api-screen.png?raw=true
    :alt: FARM Inferennce UI

One docker container exposes a REST API (localhost:5000) and another one runs a simple demo UI (localhost:3000).
You can use both of them individually and mount your own models. Check out the docs for details.


Upcoming features
###################
- More pretrained models XLNet, XLM ...
- SOTA adaptation strategies (Adapter Modules, Discriminative Fine-tuning ...)
- Enabling large scale deployment for production
- Additional Visualizations and statistics to explore and debug your model
