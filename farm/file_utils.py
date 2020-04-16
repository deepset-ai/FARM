"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
import json
import logging
import os
from functools import wraps
from hashlib import sha256
from io import open

import boto3
import numpy as np
import requests
from botocore.exceptions import ClientError
from dotmap import DotMap
from tqdm import tqdm

from transformers.file_utils import cached_path

try:
    from torch.hub import _get_torch_home

    torch_cache_home = Path(_get_torch_home())
except ImportError:
    torch_cache_home = Path(os.path.expanduser(
        os.getenv(
            "TORCH_HOME", Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "torch"
        )
    ))
default_cache_path = torch_cache_home / "farm"

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

try:
    from pathlib import Path

    FARM_CACHE = Path(os.getenv("FARM_CACHE", default_cache_path))
except (AttributeError, ImportError):
    FARM_CACHE = os.getenv("FARM_CACHE", default_cache_path)


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = FARM_CACHE

    cache_path = cache_dir / filename
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def load_from_cache(pretrained_model_name_or_path, s3_dict, **kwargs):
    # Adjusted from HF Transformers to fit loading WordEmbeddings from deepsets s3
    # Load from URL or cache if already cached
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)

    s3_file = s3_dict[pretrained_model_name_or_path]
    try:
        resolved_file = cached_path(
                        s3_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                    )

        if resolved_file is None:
            raise EnvironmentError

    except EnvironmentError:
        if pretrained_model_name_or_path in s3_dict:
            msg = "Couldn't reach server at '{}' to download data.".format(
                s3_file
            )
        else:
            msg = (
                "Model name '{}' was not found in model name list. "
                "We assumed '{}' was a path, a model identifier, or url to a configuration file or "
                "a directory containing such a file but couldn't find any such file at this path or url.".format(
                    pretrained_model_name_or_path, s3_file,
                )
            )
        raise EnvironmentError(msg)

    if resolved_file == s3_file:
        logger.info("loading file {}".format(s3_file))
    else:
        logger.info("loading file {} from cache at {}".format(s3_file, resolved_file))

    return resolved_file


def read_set_from_file(filename):
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, "r", encoding="utf-8") as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


def read_config(path):
    if path:
        with open(path) as json_data_file:
            conf_args = json.load(json_data_file)
    else:
        raise ValueError("No config provided for classifier")

    # flatten last part of config, take either value or default as value
    for gk, gv in conf_args.items():
        for k, v in gv.items():
            conf_args[gk][k] = v["value"] if (v["value"] is not None) else v["default"]

    # DotMap for making nested dictionary accessible through dot notation
    args = DotMap(conf_args, _dynamic=False)

    return args


def unnestConfig(config):
    """
    This function creates a list of config files for evaluating parameters with different values. If a config parameter
    is of type list this list is iterated over and a config object without lists is returned. Can handle lists inside any
    number of parameters.

    Can handle nested (one level) configs
    """
    nestedKeys = []
    nestedVals = []

    for gk, gv in config.items():
        if(gk != "task"):
            for k, v in gv.items():
                if isinstance(v, list):
                    if (
                        k != "layer_dims"
                    ):  # exclude layer dims, since it is already a list
                        nestedKeys.append([gk, k])
                        nestedVals.append(v)
                elif isinstance(v, dict):
                    logger.warning("Config too deep! Working on %s" %(str(v)))

    if len(nestedKeys) == 0:
        unnestedConfig = [config]
    else:
        logger.info(
            "Nested config at parameters: %s"
            % (", ".join(".".join(x) for x in nestedKeys))
        )
        unnestedConfig = []
        mesh = np.meshgrid(
            *nestedVals
        )  # get all combinations, each dimension corresponds to one parameter type
        # flatten mesh into shape: [num_parameters, num_combinations] so we can iterate in 2d over any paramter combinations
        mesh = [x.flatten() for x in mesh]

        # loop over all combinations
        for i in range(len(mesh[0])):
            tempconfig = config.copy()
            for j, k in enumerate(nestedKeys):
                if isinstance(k, str):
                    tempconfig[k] = mesh[j][
                        i
                    ]  # get ith val of correct param value and overwrite original config
                elif len(k) == 2:
                    tempconfig[k[0]][k[1]] = mesh[j][i]  # set nested dictionary keys
                else:
                    logger.warning("Config too deep! Working on %s" %(str(k)))
            unnestedConfig.append(tempconfig)

    return unnestedConfig
