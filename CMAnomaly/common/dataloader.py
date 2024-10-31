import logging
import os
import pickle
from collections import defaultdict
from glob import glob

import numpy as np

data_path_dict = {
    "industry": "./datasets/anomaly/industry",
}


def get_data_dim(dataset, subdataset=""):
    if "industry" in dataset:
        return {
            "e29ca1cd": 3,
            "c23b2b2d": 4,
            "aeb5a1de": 6,
            "2fe95315": 5,
            "0a82a873": 7,
            "af732cc4": 7,
            "b2a04b7f": 20,
            "c2970798": 13,
            "5dafb960": 12,
            "c91f4a07": 24,
            "ca2ae31d": 43,
            "f7958fb7": 37,
        }[subdataset]
    else:
        raise ValueError("unknown dataset " + str(dataset))


def load_dataset(dataset, subdataset, use_dim="all", root_dir="../", nrows=None):
    """
    use_dim: dimension used in multivariate timeseries
    """
    print("Loading {} of {} dataset".format(subdataset, dataset))
    path = data_path_dict[dataset]
    prefix = subdataset
    train_files = glob(os.path.join(root_dir, path, prefix + "_train.pkl"))
    test_files = glob(os.path.join(root_dir, path, prefix + "_test.pkl"))
    label_files = glob(os.path.join(root_dir, path, prefix + "_test_label.pkl"))

    print(os.path.join(root_dir, path, prefix + "_train.pkl"))
    logging.info("{} files found.".format(len(train_files)))

    data_dict = defaultdict(dict)

    train_data_list = []
    for idx, f_name in enumerate(train_files):
        f = open(f_name, "rb")
        train_data = pickle.load(f)
        f.close()
        if use_dim != "all":
            train_data = train_data[:, use_dim].reshape(-1, 1)
        if len(train_data) > 0:
            train_data_list.append(train_data)
    data_dict["train"] = np.concatenate(train_data_list, axis=0)[:nrows]

    test_data_list = []
    for idx, f_name in enumerate(test_files):
        f = open(f_name, "rb")
        test_data = pickle.load(f)
        f.close()
        if use_dim != "all":
            test_data = test_data[:, use_dim].reshape(-1, 1)
        if len(test_data) > 0:
            test_data_list.append(test_data)
    data_dict["test"] = np.concatenate(test_data_list, axis=0)[:nrows]

    label_data_list = []
    for idx, f_name in enumerate(label_files):
        f = open(f_name, "rb")
        label_data = pickle.load(f)
        f.close()
        if len(label_data) > 0:
            label_data_list.append(label_data)
    data_dict["test_labels"] = np.concatenate(label_data_list, axis=0)[:nrows]

    for k, v in data_dict.items():
        if k == "dim":
            continue
        print("Shape of {} is {}.".format(k, v.shape))
    return data_dict
