# -*- coding: utf-8 -*-

"""
Created on Thu Apr 03 07:30:34 2014
Updated on Fri Nov 15 00:00:00 2024

@author:
- twmeggs <twmeggs@gmail.com>
- lazuardy-tech <contact@lazuardy.tech>
"""

import os
import time

import numpy as np
from sklearn.metrics import accuracy_score

from . import anfis, membershipfunction

# start the timer
start_time = time.time()

# set the training epoch amount
epochs = 10

# load the dataset
training_set = os.getcwd() + "/lazuardy_anfis/training_set.txt"
ts = np.loadtxt(training_set, usecols=[1, 2, 3])

# extract features and target values
X = ts[:, 0:2]
Y = ts[:, 2]

# define membership functions
mf = [
    [
        ["gaussmf", {"mean": 0.0, "sigma": 1.0}],
        ["gaussmf", {"mean": -1.0, "sigma": 2.0}],
        ["gaussmf", {"mean": -4.0, "sigma": 10.0}],
        ["gaussmf", {"mean": -7.0, "sigma": 7.0}],
    ],
    [
        ["gaussmf", {"mean": 1.0, "sigma": 2.0}],
        ["gaussmf", {"mean": 2.0, "sigma": 3.0}],
        ["gaussmf", {"mean": -2.0, "sigma": 10.0}],
        ["gaussmf", {"mean": -10.5, "sigma": 5.0}],
    ],
]

# print the training information
# print(f"Data amount: {limit}")
print(f"Epoch amount: {epochs}")
# print(f"Shape of X: {X.shape}")
print(f"Number of membership function variables: {len(mf)}")

# initialize ANFIS
mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)

# train the model
anf.trainHybridJangOffLine(epochs=epochs)

# predict the output using the trained ANFIS model
predictions = anf.predict(X)

# ensure Y is binary
Y_binary = (Y > 0.5).astype(int)

# convert predictions to binary values (assuming a binary classification problem)
predictions_binary = (predictions > 0.5).astype(int)

# calculate the accuracy
accuracy = accuracy_score(Y_binary, predictions_binary)

# print accuracy
print(f"Model Accuracy on Training Data: {accuracy * 100:.2f}%")

# stop the timer
end_time = time.time()

# calculate the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# print the error plot
anf.plotErrors()

# print the result plot
anf.plotResults()
