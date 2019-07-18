import torch
from torch.utils.data import TensorDataset


# TODO we need the option to handle different dtypes
def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset
    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
    names of the type of feature and the keys are the features themselves.
    :type features: list
    :return: dataset: A PyTorch Dataset object that contains as many tensors as there are types of
    features. Each tensor will have a first dimension that is len_dataset.
    :rtype: dataset: TensorDataSet
    :return: tensor_names: The names of the different types of features
    :rtype: tensor_names: list

    """
    tensor_names = features[0].keys()
    all_tensors = []
    for t_name in tensor_names:
        cur_tensor = torch.tensor(
            [sample[t_name] for sample in features], dtype=torch.long
        )
        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names
