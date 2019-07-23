Data Handling
================================


Design Philosophy
##################
In many cases adapting a language model to your own NLP problem requires heavy work on the data preprocessing.
For training, the data needs to be read from a file, converted into training samples, featurized into vectors and finally be provided as DataLoader(s) to the model training.
For inference, the data comes not from a file, but from a request instead. The rest of the conversion looks pretty similar.


**Goals:**

* Easy adjustment to custom datasets
* Reusibility of generic pipeline components
* Clear separation of generic and dataset/task/model specific parts in the pipeline
* Powerful debugging that allows inspecting a sample in different phases of the pipeline (raw, tokenized, featurized, tensors ...)


In FARM the **Processor handles specific conversions** from files or requests to datasets. Custom datasets can be handled by extending the Processor (e.g. see CONLLProcessor).
The generic **DataSilo handles multiple datasets** (train, eval, test) and exposes DataLoaders for them. The granular classes **Sample and SampleBasket allow powerful debugging** and logging capabilities as they store different views on the same sample (raw, tokenized, featurized ...)
