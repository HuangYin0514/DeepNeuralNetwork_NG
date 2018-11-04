# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 14:48
# @Author   : HuangYin
# @FileName : lr_utils.py
# @Software : PyCharm
import h5py
import numpy as np


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(shape=(1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape(shape=(1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
