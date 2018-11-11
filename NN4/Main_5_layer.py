# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 14:44
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# load data
from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# reshape the training and test example
train_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

# Standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

# 5 Layer_Model
from NN4.L_Layer_Model import L_layer_model

layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_set_y_orig, layers_dims=layers_dims, num_iterations=2500, print_cost=True)