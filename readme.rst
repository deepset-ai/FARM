
.. image:: https://github.com/deepset-ai/FARM/blob/master/docs/img/farm_logo_text_right_wide.png?raw=true
    :width: 269
    :height: 109
    :align: left
    :alt: FARM LOGO


(**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels)

.. image:: https://travis-ci.org/deepset-ai/FARM.svg?branch=master
	:target: https://travis-ci.org/deepset-ai/FARM
	:alt: Build

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
Building upon `transformers <https://github.com/huggingface/pytorch-transformers>`_, FARM is a home for all species of pretrained language models (e.g. BERT) that can be adapted to different
**domain languages** or **down-stream tasks**.
With FARM you can easily create SOTA NLP models for tasks like document classification, NER or question answering.
The **standardized interfaces** for language models and prediction heads allow flexible extension by researchers and easy application for practitioners.
Additional experiment tracking and visualizations support you along the way to adapt a SOTA model to your own NLP problem and have a **fast proof-of-concept**.

Core features
##############
- **Easy adaptation of language models** (e.g. BERT) to your own use case
- Fast integration of **custom datasets** via Processor class
- **Modular design** of language model and prediction heads
- Switch between heads or just combine them for  **multitask learning**
- **Smooth upgrading** to new language models
- Powerful **experiment tracking** & execution
- Simple **deployment** and **visualization** to showcase your model

+------------------------------+-------------------+-------------------+-------------------+
| Task                         |      BERT         |  RoBERTa          |  XLNet            |
+==============================+===================+===================+===================+
| Text classification          | x                 |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+
| NER                          | x                 |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+
| Question Answering           | x                 |                   |                   |
+------------------------------+-------------------+-------------------+-------------------+
| Language Model Fine-tuning   | x                 |                   |                   |
+------------------------------+-------------------+-------------------+-------------------+
| Text Regression              | x                 |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+
| Multilabel Text classif.     | x                 |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+


Resources
##############
- `Full Documentation <https://farm.deepset.ai>`_
- `Intro to Transfer Learning (Blog) <https://medium.com/voice-tech-podcast/https-medium-com-deepset-ai-transfer-learning-entering-a-new-era-in-nlp-db523d9e667b>`_
- `Intro to Transfer Learning & FARM (Video) <https://www.youtube.com/watch?v=hoDgtvE-u9E&feature=youtu.be>`_
- Tutorial 1 (Overview of building blocks): `Jupyter notebook 1 <https://github.com/deepset-ai/FARM/blob/master/tutorials/1_farm_building_blocks.ipynb>`_  or `Colab 1 <https://colab.research.google.com/drive/130_7dgVC3VdLBPhiEkGULHmqSlflhmVM>`_
- Tutorial 2 (How to use custom datasets): `Jupyter notebook 2 <https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb>`_  or `Colab 2 <https://colab.research.google.com/drive/1Ce_wWu-fsy_g16jaGioe8M5mAFdLN1Yx>`_


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

.. image:: https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/code_snippet_experiment.png

*Use cases:* Training your first model, hyperparameter optimization, evaluating a language model on multiple down-stream tasks.

**Option 2: Stick together your own building blocks**

.. image:: https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/code_snippet_building_blocks.png

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

Core concepts
#########################
Model
************
AdaptiveModel = Language Model + Prediction Head(s)
With this modular approach you can easily add prediction heads (multitask learning) and re-use them for different types of language model.
`(Learn more) <https://farm.deepset.ai/modeling.html>`__


.. image:: https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/adaptive_model_no_bg_small.jpg


Data Processing
********************
Custom Datasets can be loaded by customizing the Processor. It converts "raw data" into PyTorch Datasets.
Much of the heavy lifting is then handled behind the scenes to make it fast & simple to debug.
`(Learn more) <https://farm.deepset.ai/data_handling.html>`__

.. image:: https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/data_silo_no_bg_small.jpg

Upcoming features
###################
- Improved functionality for Question Answering Task
- Additional visualizations and statistics to explore and debug your model
- More pretrained models ALBERT, XLM  ...
- Enabling large scale deployment for production

Acknowledgements
###################
- FARM is built upon parts of the great `transformers <https://github.com/huggingface/pytorch-transformers>`_  repository from Huggingface. It utilizes their implementations of the BERT model and Tokenizer.
- The original BERT model and `paper <https://arxiv.org/abs/1810.04805>`_  was published by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

Citation
###################
As of now there is no published paper on FARM. If you want to use or cite our framework, please include
the link to this repository. If you are working with the German Bert model, you can link our
`blog post <https://deepset.ai/german-bert>`_ describing its training details and performance.