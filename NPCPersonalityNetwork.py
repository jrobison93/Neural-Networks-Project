# -*- coding: utf-8 -*-
"""
@author: John Robison
"""

import math
from random import random

# The inputs are:
#   friendliness in the range of -1 to 1
#   shopOwner with the values of 0 and 1
#   importance in the range of 0 to 1
#   agressiveness in the range of 0 to 1
#   playerInCombat with the values of 0 and 1
numInputs = 5
numHiddens = 3

# The outputs are:
#   attack
#   trade
#   ignore
#   run
#   giveQuest
#   aidPlayer
numOutputs = 6

# friend, shop, importance, aggressive, inCombat,
#  [attack, trade, ignore, run, quest, aid]
training_sets = [
    [0.0, 0.0, 0.0, 0.0, 0.0, [0, 0, 1, 0, 0, 0]],
    [1.0, 1.0, 0.0, 0.0, 0.0, [0, 1, 0, 0, 0, 0]],
    [-1.0, 0.0, 0.0, 1.0, 0.0, [1, 0, 0, 0, 0, 0]],
    [-1.0, 0.0, 1.0, 1.0, 0.0, [1, 0, 0, 0, 0, 0]],
    [1.0, 0.0, 0.5, 0.5, 1.0, [0, 0, 0, 0, 0, 1]],
    [1.0, 0.0, 1.0, 0.0, 0.0, [0, 0, 0, 0, 1, 0]],
    [0.0, 0.0, 0.0, 0.0, 1.0, [0, 0, 0, 1, 0, 0]],
    [0.0, 0.0, 0.5, 0.0, 0.0, [0, 0, 0, 0, 1, 0]],
    [-1.0, 0.0, 0.0, 0.0, 1.0, [0, 0, 0, 1, 0, 0]],
    [-1.0, 1.0, 0.0, 0.0, 1.0, [0, 0, 0, 1, 0, 0]]]

output_strings = ["Attack", "Trade", "Ignore", "Run",
                  "Give Quest", "Aid Player"]

input_to_hidden = [[0.0 for x in range(numHiddens)]
                   for y in range(numInputs + 1)]
hidden_to_output = [[0.0 for x in range(numOutputs)]
                    for y in range(numHiddens + 1)]

inputs = [0.0 for x in range(numInputs)]
hiddens = [0.0 for x in range(numHiddens)]
target = [0.0 for x in range(numOutputs)]
actual = [0.0 for x in range(numOutputs)]

hidden_error = [0.0 for x in range(numHiddens)]
output_error = [0.0 for x in range(numOutputs)]

learning_rate = 0.2
number_of_epochs = 100000


# Creates a random number between -0.5 and 0.5
def randomWeight():
    return random() - 0.5


# Randomly initializes the weights of the network
def assignRandomWeights():
    for hidden in range(numHiddens):
        for inp in range(numInputs + 1):
            input_to_hidden[inp][hidden] = randomWeight()
    for hidden in range(numHiddens + 1):
        for output in range(numOutputs):
            hidden_to_output[hidden][output] = randomWeight()


# Provides the output of the sigmoid function
def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


# Provides the output of the derivative os the sigmoid function
def sigmoidDerivative(value):
    return value * (1.0 - value)


# Uses a feed forward method to determine the outputs of the network
def feedForward():
    # Computes the output of the hidden layer
    for hidden in range(numHiddens):
        sum = 0.0

        for inp in range(numInputs):
            sum += inputs[inp] * input_to_hidden[inp][hidden]

        sum += input_to_hidden[numInputs][hidden]

        hiddens[hidden] = sigmoid(sum)

    # Computes the output of the network
    for output in range(numOutputs):
        sum = 0.0

        for hidden in range(numHiddens):
            sum += hiddens[hidden] * hidden_to_output[hidden][output]

        sum += hidden_to_output[numHiddens][output]

        actual[output] = sigmoid(sum)


# Uses back propagation to calculate the error of the network
def backPropagate():
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

        hidden_error[hidden] *= sigmoidDerivative(hiddens[hidden])

    # Updates the weights of the hidden layer based on the errors
    for output in range(numOutputs):
        for hidden in range(numHiddens):
            hidden_to_output[hidden][output] += (learning_rate *
                                                 output_error[output] *
                                                 hiddens[hidden])

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


def action():
    index = 0

    maximum = actual[index]

    for i in range(numOutputs):
        if(actual[i] > maximum):
            maximum = actual[i]
            index = i

    return index

statistic_file = open("statistics.csv", "w")
weights_file = open("weights.txt", "w")

assignRandomWeights()
weights_file.write("Initial Network\n")
weights_file.write("---------------\n")
for i in input_to_hidden:
    weights_file.write(str(i) + " ")
weights_file.write("\n")
for i in hidden_to_output:
    weights_file.write(str(i) + " ")
weights_file.write("\n")


sample = 0
max_samples = len(training_sets) - 1

# Trains the neural network with the training set
for i in range(number_of_epochs + 1):
    inputs = training_sets[sample][0:5]
    target = training_sets[sample][5]

    feedForward()

    error = 0.0
    for j in range(numOutputs):
        error += (training_sets[sample][5][j] - actual[j]) ** 2

    error = 0.5 * error

    if i % 50 == 0:
        statistic_file.write(str(i) + ", " + str(error) + "\n")
        print(str(i), ", " + str(error))

    backPropagate()

    if sample == max_samples:
        sample = 0
    else:
        sample += 1


weights_file.write("Final Network\n")
weights_file.write("---------------\n")
for i in input_to_hidden:
    weights_file.write(str(i) + " ")
weights_file.write("\n")
for i in hidden_to_output:
    weights_file.write(str(i) + " ")
weights_file.write("\n")


statistic_file.close()
weights_file.close()
