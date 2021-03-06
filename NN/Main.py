# -*- coding: utf-8 -*-
# @Time     : 2018/11/2 20:40
# @Author   : HuangYin
# @FileName : Main.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
from testCases import linear_forward_test_case, linear_activation_forward_test_case, L_model_forward_test_case, \
    compute_cost_test_case, linear_backward_test_case, linear_activation_backward_test_case, L_model_backward_test_case, \
    update_parameters_test_case

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
from L_Model_Forward import L_model_forward

X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

# compute cost
from Compute_Cost import comput_cost

Y, AL = compute_cost_test_case()
cost = comput_cost(AL, Y)
print("cost = " + str(cost))

# linear_backforward propagation
from Linear_Backward import linear_backward

dZ, cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, cache)
print("dA_prev = " + str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))

# linear_activation backward
from Linear_Activation_Backward import linear_activation_backward

AL, linear_activation_cache = linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, "sigmoid")
print("sigmoid:")
print("dA_prev =" + str(dA_prev))
print("dW =" + str(dW))
print("db =" + str(db))
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, "relu")
print("relu:")
print("dA_prev =" + str(dA_prev))
print("dW =" + str(dW))
print("db =" + str(db))

# L_model_backward
from L_Model_Backward import L_model_Backward

AL, Y, caches = L_model_backward_test_case()
grads = L_model_Backward(AL, Y, caches)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dA1 = " + str(grads["dA1"]))

# update parameters
from Update_Parameters import update_parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


