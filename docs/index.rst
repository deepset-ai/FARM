.. FARM documentation master file, created by
   sphinx-quickstart on Wed Jul  3 18:08:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FARM!
================================

**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels

What is it?
#############
FARM makes cutting edge **Transfer Learning** for NLP simple.
It is a home for all species of pretrained language models (e.g. BERT) that can be adapted to different down-stream
tasks (e.g. NER) by simply switching the prediction head.
The standardized interfaces for language models and prediction heads allow a flexible extension by researchers and an easy adaptation for practitioners.
Additional experiment tracking and visualizations support you along the way to adapt a SOTA model to your own NLP problem and showcase it as a PoC.

Core features
###############
- Easy adaptation of pretrained language models (e.g. BERT) to your own use case
- Modular design of language model and prediction heads
- Easy experiment tracking & execution
- Simple deployment and visualization to showcase your PoC


Upcoming features
###################
- More pretrained models XLNet, XLM ...
- SOTA adaptation strategies (Adapter Modules, Discriminative Fine-tuning ...)
- Enabling large scale deployment for production
- Additional Visualizations and statistics to explore and debug your model


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   basic_usage
   examples



.. toctree::
   :maxdepth: 1
   :caption: Concepts

   data_handling
   modeling


.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/data_handling
   api/modeling
   api/running


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
