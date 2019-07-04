import torch
from torch.utils.data import TensorDataset


# TODO: Make this more generic somehow
# def convert_features_to_dataset(features, label_dtype=torch.long):
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     all_label_ids = torch.tensor([f.label_id for f in features], dtype=label_dtype)
#     try:
#         all_initial_masks = torch.tensor(
#             [f.initial_mask for f in features], dtype=torch.long
#         )
#     except (AttributeError, TypeError):
#         all_initial_masks = torch.tensor([0] * len(features), dtype=torch.long)
#
#     dataset = TensorDataset(
#         all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_initial_masks
#     )
#     return dataset

# TODO we need the option to handle different dtypes
def convert_features_to_dataset(features, label_dtype=torch.long):
    feature_names = features[0].features.keys()
    all_tensors = []
    for f_name in feature_names:
        cur_tensor = torch.tensor(
            [sample.features[f_name] for sample in features], dtype=torch.long
        )
        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset
