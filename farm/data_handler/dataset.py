import numpy as np
import numbers
import logging
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


# TODO we need the option to handle different dtypes
def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs (e.g. Natural Questions downsamples
    # instances of is_impossible
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        # Conversion of floats
        if t_name == 'regression_label_ids':
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)
        else:
            # Checking weather a non-integer will be silently converted to torch.long
            try:
                if isinstance(features[0][t_name], numbers.Number):
                    basenum = features[0][t_name]
                elif isinstance(features[0][t_name], list):
                    basenum = features[0][t_name][0]
                else:
                    basenum = features[0][t_name].ravel()[0]
            except:
                basenum = features[0][t_name]
            if not np.issubdtype(type(basenum), np.integer):
                logger.warning(f"Problem during conversion to torch tensors:\n"
                               f"A non-integer value for '{t_name}' with a value of: "
                               f"'{basenum}' will be converted to a torch tensor of dtype long.")

            # Convert all remaining python objects to torch long tensors
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names
