from torch.utils.data import DataLoader
import torch


class NamedDataLoader(DataLoader):
    def __init__(self, dataset, sampler, batch_size, tensor_names):

        def collate_fn(batch):
            assert len(batch[0]) == len(tensor_names), \
                "Dataset contains {} tensors while there are {} tensor names supplied: {}".format(len(batch[0]),
                                                                                                  len(tensor_names),
                                                                                                  tensor_names)
            lists_temp = [[] for _ in range(len(tensor_names))]
            ret = dict(zip(tensor_names, lists_temp))

            for example in batch:
                for name, tensor in zip(tensor_names, example):
                    ret[name].append(tensor)

            for key in ret:
                ret[key] = torch.stack(ret[key])

            return ret


        super(NamedDataLoader, self).__init__(dataset=dataset,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)


# class NamedDataLoader(DataLoader):
#     def __init__(self, names ...):
#
#     def collate(self, names):
#         ....
#


def covert_dataset_to_dataloader(dataset, sampler, batch_size):
    sampler_initialized = sampler(dataset)
    data_loader = DataLoader(
        dataset, sampler=sampler_initialized, batch_size=batch_size
    )
    return data_loader
