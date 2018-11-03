# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 20:40
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
from testCases import linear_forward_test_case, linear_activation_forward_test_case, L_model_forward_test_case

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 使每次随机生成数值相同
# 仅生效一次
np.random.seed(1)

# Initialize parameters
from Initialize_Parameters import initialize_parameters

parameters = initialize_parameters(2, 2, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

from Initialize_Parameters_Deep import initialize_parameters_deep

parameters = initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# Forward propagation
from Linear_Forward import linear_forward

A, W, b = linear_forward_test_case()
Z, cache = linear_forward(A, W, b)
print("Z = " + str(Z))
print("A = " + str(cache[0]))
print("W = " + str(cache[1]))
print("b = " + str(cache[2]))

# forward propagation activation function
from Linear_Activation_Forward import linear_activation_forward

A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, "sigmoid")
print("With sigmoid : A = " + str(A))
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, "relu")
print("With relu :A = " + str(A))

# L_model_test
