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
    resource.setrlimit(resource.RLIMIT_NOFILE, (10_000, rlimit[1]))
