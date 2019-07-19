# FARM

(**F**ramework for **A**dapting **R**epresentation **M**odels)  

## What is it?
FARM makes cutting edge **Transfer Learning** for NLP simple. 
It is a home for all species of pretrained language models (e.g. BERT) that can be adapted to different down-stream
tasks (e.g. NER) by simply switching the prediction head.
The standardized interfaces for language models and prediction heads allow a flexible extension by researchers and an easy adaptation for practitioners.
Additional experiment tracking and visualizations support you along the way to adapt a SOTA model to your own NLP problem and showcase it as a PoC.  

## Core features  

- Easy adaptation of pretrained language models (e.g. BERT) to your own use case
- Modular design of language model and prediction heads
- Easy Experiment tracking & execution
- Simple deployment and visualization to showcase your PoC model 


## How to use it?

### REST APIs for downstream tasks with Docker
The package contains a Docker image with a Flask App for serving HTTP APIs to infer downstream tasks on trained models.

To build and run Docker container from source:

* Run `docker build . -t farm-api` to build the image.
* Download the models from the link or use your own generated using the framework. For NER task, run the following command to download 
and extract model: `wget -qO- https://farm-public.s3.eu-central-1.amazonaws.com/bert-german-CONLL2003.tar.gz | tar xvz - - -C <save-dir-path>`.
* Run `docker run -v </save-dir-path>:/home/user/save farm-api` to run a container with the model directory mounted as a docker volume. 


### Quickstart

### Train a model

### Start Model API

### Showcase UI

## Vision & upcoming features
- Adding XLNet
- SOTA adaptation strategies (Adapter Modules, Discriminative Fine-tuning ...)
- Enabling large scale deployment for production
- Additional Visualizations and statistics to explore and debug your model