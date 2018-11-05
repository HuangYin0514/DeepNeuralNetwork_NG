# -*- coding: utf-8 -*-
# @Time     : 2018/11/4 20:06
# @Author   : HuangYin
# @FileName : L_Layer_Model.py
# @Software : PyCharm

from Initialize_Parameters_Deep import initialize_parameters_deep
from L_Model_Forward import L_model_forward
from L_Model_Backward import L_model_Backward
from Compute_Cost import comput_cost
from Update_Parameters import update_parameters
import matplotlib.pyplot as plt
import numpy as np
from Linear_Activation_Forward import linear_activation_forward
from Linear_Activation_Backward import linear_activation_backward


def L_layer_model_nonFor(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    parameters = initialize_parameters_deep(layers_dims)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    costs = []
    grads = {}
    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = comput_cost(A2, Y)
        if print_cost and i % 100 == 0:
            costs.append(cost)
            print("Cost after iterations {}:{}".format(i, cost))
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2


        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

    return parameters
