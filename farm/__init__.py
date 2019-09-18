import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


#fixing a environment issue on increasing number file descriptors
#fix accoring to this pytorch issue: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1_000_000, rlimit[1]))