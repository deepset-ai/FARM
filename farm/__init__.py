import logging

import torch.multiprocessing as mp

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
if "file_descriptor" in mp.get_all_sharing_strategies():
    import resource

    mp.set_sharing_strategy("file_descriptor")

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # seting soft limit to hard limit (=rlimit[1]) minus a small amount to be safe
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1]-512, rlimit[1]))
