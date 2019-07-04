from torch.utils.data import DataLoader


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
