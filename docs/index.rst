.. FARM documentation master file, created by
   sphinx-quickstart on Wed Jul  3 18:08:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FARM!
================================

**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels

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
