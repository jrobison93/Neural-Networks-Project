# -*- coding: utf-8 -*-
"""
@author: John Robison
"""

import math

numInputs = 
numHiddens = 
numOutputs = 

input_to_hidden = [][]
hidden_to_output = [][]

inputs = []
hidden = []
target = []
actual = []

hidden_error = []
output_error = []

learning_rate = 0.2
number_of_epochs = 10000


# Creates a random number between -0.5 and 0.5
def randomWeight():
    return random.random() - 0.5


# Randomly initializes the weights of the network
def assignRandomWeights():
    for hidden in range(numHiddens):
        for inp in range(numInputs):
            input_to_hidden[inp][hidden] = randomWeight()
        for output in range(numOutputs):
            hidden_to_output[hidden][output] = randomWeight()


# Provides the output of the sigmoid function
def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-val))


# Provides the output of the derivative os the sigmoid function
def sigmoidDerivative(value):
    return value * (1.0 - value)


# Uses a feed forward method to determine the outputs of the network
def feedForward:
    # Computes the output of the hidden layer
    for hidden in range(numHiddens):
        sum = 0.0

        for inp in range(numInputs):
            sum += inputs[inp] * inputs_to_hidden[inp][hidden]

        sum += input_to_hidden[numInputs][hidden]

        hidden[hidden] = sigmoid(sum)

    # Computes the output of the network
    for output in range(numOutputs):
        sum = 0.0

        for hidden in range(numHiddens):
            sum += hidden[hidden] * hidden_to_output[hidden][output]

        sum += hidden_to_output[numHiddens][output]

        actual[output] = sigmoid(sum)


# Uses back propagation to calculate the error of the network
def backPropagate:
    # Calculates the error of the output layer
    for output in range(numOutputs):
        output_error[output] = ((target[output] - actual[output]) *
                                sigmoidDerivative(actual[output]))

    # Calculates the error of the hidden layer
    for hidden in range(numHiddens):
        hidden_error[hidden] = 0.0

        for output in range(numOutputs):
            hidden_error[hidden] += (output_error[output] *
                                     hidden_to_output[hidden][output])

        hidden_error[hidden] *= sigmoidDerivative(hidden[hidden])

    # Updates the weights of the hidden layer based on the errors
    for output in range(numOutputs):
        for hidden in range(numHiddens):
            hidden_to_output[hidden][output] += (learning_rate *
                                                 output_error[output] *
                                                 hidden[hidden])

        hidden_to_output[numHiddens][output] += (learning_rate *
                                                 output_error[output])

    # Updates the weights of the output layer based on the errors
    for hidden in range(numHiddens):
        for inp in range(numInputs):
            input_to_hidden[inp][hidden] += (learning_rate *
                                             hidden_error[hidden] *
                                             inputs[inp])

        input_to_hidden[numInputs][hidden] += (learning_rate *
                                               hidden_error[hidden])

statistic_file = open("statistics.txt", "w")
