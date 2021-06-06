# Diary of work

Started this on 2021-06-06 since it got really hard to keep track after all the work already done.


## 2021-06-06

* resume work on issue 408, PR778
* pulled upstream commits to our fork
* pulled into our local master
* merge master into the issue408 branch
* re-test the example script if it still works: run FARM `examples/mtl01_tclass_tclass.py`
  * dev evaluation after 840 batches 
  * coarse: loss=0.88439, f1macro=0.79643
  * fine: loss=1.20133 f1macro=0.53625
  * test evaluation after 846 batches:
  * coarse: loss=1.2634, f1macro=0.7569
  * fine: loss=1.57499, f1macro=0.480277
  * ???? why are these evaluations so similar: if there is a test set, farm automatically does a test evaluation
    in addition to dev evaluation, then we do the test evaluaton manually again, which really is not needed.
    We could just replace with training set resubstitution estimate to see how this compares to generalization
    Updated numbers above for proper dev/test runs
  * !! Got exception when exiting: "AttributeError: 'NoneType' object has no attribute 'dumps'" in queues.py 362
* change the example script to do round-robin training instead of parallel training and compare result
  * see https://github.com/deepset-ai/FARM/pull/220
  * RR dev set evaluation after 840 batches:
    * coarse: loss=0.881459 f1macro=0.814499
    * fine: loss=1.20056 f1macro=0.54793
  * dev set  eval after 1680 batches:python 
    * coarse: loss=0.99316 f1macro=0.8187
    * fine: loss=1.36716 f1macro=0.5845
  * test set eval after 1692 batches:
    * coarse: loss=1.4100 f1macro=0.73411
    * fine: loss=1.7741 f1macro=0.4631
  * train set eval:
    * coarse: f1macro=0.9992
    * fine: f1macro=0.9965
* Change things so that number of epochs does NOT get doubled for round-robin!
* coarse alone:

* now need to address  the missing task name in the inference return data

Understanding what is going on:

Class Processor 
* `processor.add_task(name=taskname,task_type="classification",...)` adds dictionary as `processor.tasks[taskname]`
  * added: set key `task_name` to name in the dict
* `Processor(..., tasks=somedict)` expects a task dict with key taskname.
  * added: make shallow copy of passed task dicts
  * added: add `task_name` to each copied dict
  
Classes XxxHead:
* paramater `task_name` is passed to constructor  and stored in `self.task_name`
* methods formatted preds should include the task name
