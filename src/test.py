# -*- coding: utf-8 -*-

"""
Created on Thu Apr 03 07:30:34 2014
Updated on Fri Nov 15 00:00:00 2024

@author:
- twmeggs <twmeggs@gmail.com>
- lazuardy-tech <contact@lazuardy.tech>
"""

import numpy

import anfis
import membershipfunction
import mfderivs

# define the training_set.txt file path, please change it to your own path
training_set = "/Users/ezra/projects/lazuardy/oss/anfis/src/training_set.txt"

ts = numpy.loadtxt(training_set, usecols=[1, 2, 3])
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
