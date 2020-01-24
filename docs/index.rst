.. FARM documentation master file, created by
   sphinx-quickstart on Wed Jul  3 18:08:08 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FARM!
================================

**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels

What is it?
############
FARM makes **Transfer Learning** with BERT & Co **simple, fast and enterprise-ready**.
It's build upon `transformers <https://github.com/huggingface/pytorch-transformers>`_ and provides additional features to simplify the life of developers:
Parallelized preprocessing, highly modular design, multi-task learning, experiment tracking, easy debugging and close integration with AWS SageMaker.

With FARM you can build **fast proof-of-concepts** for tasks like text classification, NER or question answering and **transfer them easily into production**.


- `What is it? <https://github.com/deepset-ai/FARM#what-is-it>`_
- `Core Features <https://github.com/deepset-ai/FARM#core-features>`_
- `Resources <https://github.com/deepset-ai/FARM#resources>`_
- `Installation <https://github.com/deepset-ai/FARM#installation>`_
- `Basic Usage <https://github.com/deepset-ai/FARM#basic-usage>`_
- `Advanced Usage <https://github.com/deepset-ai/FARM#advanced-usage>`_
- `Core Concepts <https://github.com/deepset-ai/FARM#core-concepts>`_
- `FAQ <https://github.com/deepset-ai/FARM#faq>`_
- `Upcoming features <https://github.com/deepset-ai/FARM#upcoming-features>`_


Core features
##############
- **Easy fine-tuning of language models** to your task and domain language
- **Speed**: AMP optimizers (~35% faster) and parallel preprocessing (16 CPU cores => ~16x faster)
- **Modular design** of language model and prediction heads
- Switch between heads or just combine them for **multitask learning**
- **Full Compatibility** with transformers' models and model hub
- **Smooth upgrading** to newer language models
- Integration of **custom datasets** via Processor class
- Powerful **experiment tracking** & execution
- **Checkpointing & Caching** to resume training and reduce costs with spot instances
- Simple **deployment** and **visualization** to showcase your model

+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Task                         |      BERT         |  RoBERTa          |  XLNet            |  ALBERT           |  DistilBERT       |  XLMRoBERTa       |
+==============================+===================+===================+===================+===================+===================+===================+
| Text classification          | x                 |  x                |  x                |  x                |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| NER                          | x                 |  x                |  x                |  x                |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Question Answering           | x                 |  x                |  x                |  x                |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Language Model Fine-tuning   | x                 |                   |                   |                   |                   |                   |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Text Regression              | x                 |  x                |  x                |  x                |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Multilabel Text classif.     | x                 |  x                |  x                |  x                |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Extracting embeddings        | x                 |  x                |  x                |  x                |  x                |  x                |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| LM from scratch (beta)       | x                 |                   |                   |                   |                   |                   |
+------------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+


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
