"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import fnmatch
import json
import logging
import os
import shutil
import sys
import tempfile
from functools import wraps
from hashlib import sha256
from io import open

import boto3
import numpy as np
import requests
from botocore.exceptions import ClientError
from dotmap import DotMap
from tqdm import tqdm

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv(
            "TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch")
        )
    )
default_cache_path = os.path.join(torch_cache_home, "farm")

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
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
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


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = FARM_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


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


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = FARM_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        try:
            response = requests.head(url, allow_redirects=True)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get("ETag")
        except EnvironmentError:
            etag = None

    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode("utf-8")
    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + ".*")
        matching_files = list(filter(lambda s: not s.endswith(".json"), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w") as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(
                        output_string, "utf-8"
                    )  # The beauty of python 2
                meta_file.write(output_string)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


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


def read_config(path, flattend=False):
    if path:
        with open(path) as json_data_file:
            conf_args = json.load(json_data_file)
    else:
        raise ValueError("No config provided for classifier")

    def getArgValue(arg):
        if "value" not in arg:
            logger.error(
                "Only depth 2 config files supported. Failed to convert: %s" % str(arg)
            )
        return arg["value"] if (arg["value"] is not None) else arg["default"]

    # flatten last part of config, take either value or default as value
    for gk, gv in conf_args.items():
        for k, v in gv.items():
            if isinstance(getArgValue(v), dict):
                logger.error("Config is too deeply nested, at %s" % str(v))
            conf_args[gk][k] = getArgValue(v)

    # DotMap for making nested dictionary accessible through dot notation
    flat_args = dict(
        conf_args["general"],
        **conf_args["task"],
        **conf_args["parameter"],
        **conf_args["logging"],
    )
    if flattend:
        args = DotMap(flat_args, _dynamic=False)
    else:
        args = DotMap(conf_args, _dynamic=False)

    return args


def unnestConfig(config, flattened=False):
    """
    This function creates a list of config files for evaluating parameters with different values. If a config parameter
    is of type list this list is iterated over and a config object without lists is returned. Can handle lists inside any
    number of parameters.

    Can handle shallow or nested (one level) configs
    """
    nestedKeys = []
    nestedVals = []
    if flattened:
        for k, v in config.items():
            if isinstance(v, list):
                if k != "layer_dims":  # exclude layer dims, since it is already a list
                    nestedKeys.append(k)
                    nestedVals.append(v)
    else:
        for gk, gv in config.items():
            if(gk != "task"):
                for k, v in gv.items():
                    if isinstance(v, list):
                        if isinstance(v, list):
                            if (
                                k != "layer_dims"
                            ):  # exclude layer dims, since it is already a list
                                nestedKeys.append([gk, k])
                                nestedVals.append(v)
                        elif isinstance(v, dict):
                            logger.error("Config too deep!")

    if len(nestedKeys) == 0:
        unnestedConfig = [config]
    else:
        if flattened:
            logger.info("Nested config at parameters: %s" % (", ".join(nestedKeys)))
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
                    logger.error("Config too deep!")
            unnestedConfig.append(tempconfig)

    return unnestedConfig
