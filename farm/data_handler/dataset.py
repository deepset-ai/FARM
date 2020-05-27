import torch
from torch.utils.data import TensorDataset


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
        try:
            cur_tensor = torch.tensor(
                [sample[t_name] for sample in features], dtype=torch.long
            )
        except ValueError:
            cur_tensor = torch.tensor(
                [sample[t_name] for sample in features], dtype=torch.float32
            )

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names
