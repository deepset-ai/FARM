from torch.utils.data import DataLoader


def covert_dataset_to_dataloader(dataset, sampler, batch_size):
    sampler_initialized = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler_initialized, batch_size=batch_size)
    return data_loader
