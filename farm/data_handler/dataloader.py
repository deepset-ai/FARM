from torch.utils.data import DataLoader, Dataset, Sampler
import torch


class NamedDataLoader(DataLoader):
    """
    A modified version of the PyTorch DataLoader that returns a dictionary where the key is
    the name of the tensor and the value is the tensor itself
    """

    def __init__(self, dataset, sampler, batch_size, tensor_names):
        """
        :param dataset: The dataset that will be wrapped by this NamedDataLoader
        :type dataset: Dataset
        :param sampler: The sampler used by the NamedDataLoader to choose which samples to include in the batch
        :type Sampler
        :param batch_size: The size of the batch to be returned by the NamedDataLoader
        :type int
        :param tensor_names: The names of the tensor, in the order that the dataset returns them in.
        :type tensor_names: list
        """

        def collate_fn(batch):
            """
            A custom collate function that formats the batch as a dictionary where the key is
            the name of the tensor and the value is the tensor itself
            """
            assert len(batch[0]) == len(
                tensor_names
            ), "Dataset contains {} tensors while there are {} tensor names supplied: {}".format(
                len(batch[0]), len(tensor_names), tensor_names
            )
            lists_temp = [[] for _ in range(len(tensor_names))]
            ret = dict(zip(tensor_names, lists_temp))

            for example in batch:
                for name, tensor in zip(tensor_names, example):
                    ret[name].append(tensor)

            for key in ret:
                ret[key] = torch.stack(ret[key])

            return ret

        super(NamedDataLoader, self).__init__(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )


def covert_dataset_to_dataloader(dataset, sampler, batch_size):
    sampler_initialized = sampler(dataset)
    data_loader = DataLoader(
        dataset, sampler=sampler_initialized, batch_size=batch_size
    )
    return data_loader
