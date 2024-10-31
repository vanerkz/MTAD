import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class preprocessor:
    def __init__(self):
        self.vocab_size = None
        self.discretizer_list = defaultdict(list)

    def save(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        print("Saving preprocessor into {}".format(filepath))
        with open(filepath, "wb") as fw:
            pickle.dump(self.__dict__, fw)

    def load(self, filepath):
        filepath = os.path.join(filepath, "preprocessor.pkl")
        print("Loading preprocessor from {}".format(filepath))
        with open(filepath, "rb") as fw:
            self.__dict__.update(pickle.load(fw))

    def normalize(self, data_dict, method="minmax"):
        print("Normalizing data")
        # method: minmax, standard, robust
        normalized_dict = defaultdict(dict)

        # fit_transform using train
        if method == "minmax":
            est = MinMaxScaler(clip=True)
        elif method == "standard":
            est = StandardScaler()
        elif method == "robust":
            est = RobustScaler()

        train_ = est.fit_transform(data_dict["train"])
        test_ = est.transform(data_dict["test"])

        # assign back
        normalized_dict["train"] = train_
        normalized_dict["test"] = test_

        for k, v in data_dict.items():
            if k not in ["train", "test"]:
                normalized_dict[k] = v
        return normalized_dict


def get_windows(ts, labels=None, window_size=128, stride=1, dim=None):
    i = 0
    ts_len = ts.shape[0]
    windows = []
    label_windows = []
    while i + window_size < ts_len:
        if dim is not None:
            windows.append(ts[i : i + window_size, dim])
        else:
            windows.append(ts[i : i + window_size])
        if labels is not None:
            label_windows.append(labels[i : i + window_size])
        i += stride
    if labels is not None:
        return np.array(windows, dtype=np.float32), np.array(
            label_windows, dtype=np.float32
        )
    else:
        return np.array(windows, dtype=np.float32), None


def generate_windows_with_index(
    data_dict,
    data_hdf5_path=None,
    window_size=100,
    nrows=None,
    clear=False,
    stride=1,
    **kwargs
):

    results = {}


    print("Generating sliding windows (size {}).".format(window_size))

    if "train" in data_dict:
        train = data_dict["train"][0:nrows]
        train_windows, _ = get_windows(train, window_size=window_size, stride=stride)

    if "test" in data_dict:
        test = data_dict["test"][0:nrows]
        test_label = (
            None
            if "test_labels" not in data_dict
            else data_dict["test_labels"][0:nrows]
        )
        test_windows, test_labels = get_windows(
            test, test_label, window_size=window_size, stride=1
        )

    if len(train_windows) > 0:
        results["train_windows"] = train_windows
        print("Train windows #: {}".format(train_windows.shape))

    if len(test_windows) > 0:
        if test_label is not None:
            results["test_windows"] = test_windows
            results["test_labels"] = test_labels
        else:
            results["test_windows"] = test_windows
        print("Test windows #: {}".format(test_windows.shape))

    idx = np.asarray(list(range(0, test.shape[0] + stride * window_size)))
    i = 0
    ts_len = test.shape[0]
    windows = []
    while i + window_size < ts_len:
        windows.append(idx[i : i + window_size])
        i += 1

    index = np.array(windows)

    results["index_windows"] = index

    # save_hdf5(cache_file, results)
    return results


def generate_windows(
    data_dict,
    use_token=False,
    window_size=100,
    nrows=None,
    clear=False,
    stride=1,
    test_stride=1,
    **kwargs
):
    results = {}

    print("Generating sliding windows (size {}, stride {}, test stride {}).".format(window_size, stride, test_stride))

    if "train" in data_dict:
        if use_token:
            train = data_dict["train_tokens"][0:nrows]
        else:
            train = data_dict["train"][0:nrows]

        train_windows, _ = get_windows(train, window_size=window_size, stride=stride)

    if "test" in data_dict:
        if use_token:
            test = data_dict["test_tokens"][0:nrows]
        else:
            test = data_dict["test"][0:nrows]
        test_label = (
            None
            if "test_labels" not in data_dict
            else data_dict["test_labels"][0:nrows]
        )
        test_windows, test_labels = get_windows(
            test, test_label, window_size=window_size, stride=test_stride
        )

    if len(train_windows) > 0:
        results["train_windows"] = train_windows
        print("Train windows #: {}".format(train_windows.shape))

    if len(test_windows) > 0:
        if test_label is not None:
            results["test_windows"] = test_windows
            results["test_labels"] = test_labels
        else:
            results["test_windows"] = test_windows
        print("Test windows #: {}".format(test_windows.shape))

    # save_hdf5(cache_file, results)
    return results
