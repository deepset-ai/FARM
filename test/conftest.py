import pytest

from farm.infer import Inferencer
import boto3
import os


@pytest.fixture(scope="module")
def adaptive_model_qa():
    # download the model from S3
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("deepset.ai-farm-models")
    prefix = "0.3.0/bert-english-qa-large/"
    for object in bucket.objects.filter(Prefix=prefix):
        if not os.path.exists(os.path.dirname(object.key)):
            os.makedirs(os.path.dirname(object.key))
        bucket.download_file(object.key, object.key)
    model = Inferencer.load(prefix, batch_size=16)
    return model
