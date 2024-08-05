from .credit import Credit

dataset_list = {
                "credit": Credit
                }


def build_dataset(dataset, root_path, shots, preprocess):
    return dataset_list[dataset](root_path, shots)