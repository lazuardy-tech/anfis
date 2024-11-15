# -*- coding: utf-8 -*-

"""
Created on Thu Apr 03 07:30:34 2014
Updated on Fri Nov 15 00:00:00 2024

@author:
- twmeggs <twmeggs@gmail.com>
- lazuardy-tech <contact@lazuardy.tech>
"""

import os

import numpy as np

from . import anfis, membershipfunction

# define the training_set.txt file path
training_set = os.getcwd() + "/lazuardy_anfis/training_set.txt"

ts = np.loadtxt(training_set, usecols=[1, 2, 3])
X = ts[:, 0:2]
Y = ts[:, 2]

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


mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=20)
print(round(anf.consequents[-1][0], 6))
print(round(anf.consequents[-2][0], 6))
print(round(anf.fittedValues[9][0], 6))
if (
    round(anf.consequents[-1][0], 6) == -5.275538
    and round(anf.consequents[-2][0], 6) == -1.990703
    and round(anf.fittedValues[9][0], 6) == 0.002249
):
    print("test is good")

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
