FROM deepset/farm-gpu:latest
COPY examples examples
#COPY data/test data/test

# ENV SAGEMAKER_PROGRAM train.py
ENTRYPOINT ["python3","-m", "torch.distributed.launch", "--nproc_per_node=4", "examples/train_from_scratch_with_sagemaker.py"]
