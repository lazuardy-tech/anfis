import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skfuzzy import gaussmf, gbellmf, partial_dmf, sigmf


class ANFIS:
    """Class to implement an Adaptive Network Fuzzy Inference System: ANFIS"""

    def __init__(self, X, Y, memFunction):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.tensor(copy.copy(X), dtype=torch.float32).to(self.device)
        self.Y = torch.tensor(copy.copy(Y), dtype=torch.float32).to(self.device)
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [
            [x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))
        ]
        self.rules = torch.tensor(list(itertools.product(*self.memFuncsByVariable))).to(
            self.device
        )
        self.consequents = torch.zeros(
            self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1), dtype=torch.float32
        ).to(self.device)
        self.errors = torch.empty(0, dtype=torch.float32).to(self.device)
        self.memFuncsHomo = all(
            len(i) == len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable
        )
        self.trainingType = "Not trained yet"

    def LSE(self, A, B, initialGamma=1000.0):
        coeffMat = (
            A
            if isinstance(A, torch.Tensor)
            else torch.tensor(A, dtype=torch.float32).to(self.device)
        )
        rhsMat = (
            B
            if isinstance(B, torch.Tensor)
            else torch.tensor(B, dtype=torch.float32).to(self.device)
        )
        S = torch.eye(coeffMat.shape[1], device=self.device) * initialGamma
        x = torch.zeros((coeffMat.shape[1], 1), device=self.device)

        for i in range(len(coeffMat[:, 0])):
            a = coeffMat[i, :].unsqueeze(0)
            b = rhsMat[i].unsqueeze(0)

            # Ensure a and S have compatible dimensions
            if a.shape[1] != S.shape[0]:
                a = a.permute(1, 0)

            S = S - (S @ a.mT @ a @ S) / (1 + (a @ S @ a.mT))
            x = x + S @ a.mT @ (b - a @ x)

        return x

    def trainHybridJangOffLine(
        self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01
    ):
        self.trainingType = "trainHybridJangOffLine"
        convergence = False
        epoch = 1

        while (epoch < epochs) and (not convergence):
            # layer four: forward pass
            [layerFour, wSum, w] = self.forwardHalfPass(self.X)

            # layer five: least squares estimate
            layerFive = self.LSE(layerFour, self.Y, initialGamma)
            self.consequents = layerFive
            layerFive = layerFour @ layerFive

            # error
            error = torch.sum((self.Y - layerFive.T) ** 2)
            print("current error: " + str(error.item()))
            average_error = torch.mean(torch.abs(self.Y - layerFive.T))
            self.errors = torch.cat(
                [self.errors, torch.tensor([error], device=self.device)]
            )

            if len(self.errors) != 0:
                if self.errors[-1] < tolerance:
                    convergence = True

            # back propagation
            if not convergence:
                cols = range(self.X.shape[1])
                dE_dAlpha = [
                    self.backprop(colX, cols, wSum, w, layerFive)
                    for colX in range(self.X.shape[1])
                ]

            if len(self.errors) >= 4:
                if (
                    self.errors[-4]
                    > self.errors[-3]
                    > self.errors[-2]
                    > self.errors[-1]
                ):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (
                    (self.errors[-1] < self.errors[-2])
                    and (self.errors[-3] < self.errors[-2])
                    and (self.errors[-3] < self.errors[-4])
                    and (self.errors[-5] > self.errors[-4])
                ):
                    k = k * 0.9

            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            eta = k / torch.abs(torch.sum(torch.tensor(t, device=self.device)))

            if torch.isinf(eta):
                eta = k

            dAlpha = copy.deepcopy(dE_dAlpha)
            if not self.memFuncsHomo:
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * torch.tensor(
                    [
                        item
                        for sublist in dE_dAlpha
                        for subsublist in sublist
                        for item in subsublist
                    ],
                    device=self.device,
                ).view(len(dE_dAlpha), len(dE_dAlpha[0]), len(dE_dAlpha[0][0]))

            for varsWithMemFuncs in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[varsWithMemFuncs])):
                    paramList = sorted(self.memFuncs[varsWithMemFuncs][MFs][1])
                    for param in range(len(paramList)):
                        self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]] = (
                            self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]]
                            + dAlpha[varsWithMemFuncs][MFs][param].item()
                        )
            epoch += 1

        self.fittedValues = self.predict(self.X)
        self.residuals = self.Y - self.fittedValues[:, 0]

        return self.fittedValues

    def plotMF(self, x, inputVar):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        for mf in range(len(self.memFuncs[inputVar])):
            if self.memFuncs[inputVar][mf][0] == "gaussmf":
                y = gaussmf(x.cpu().numpy(), **self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == "gbellmf":
                y = gbellmf(x.cpu().numpy(), **self.memClass.MFList[inputVar][mf][1])
            else:
                y = sigmf(x.cpu().numpy(), **self.memClass.MFList[inputVar][mf][1])

            plt.plot(x.cpu().numpy(), y, "r")

        plt.show()

    def plotErrors(self):
        if self.trainingType == "Not trained yet":
            print(self.trainingType)
        else:
            plt.plot(
                range(len(self.errors)), self.errors.cpu().numpy(), "ro", label="errors"
            )
            plt.ylabel("error")
            plt.xlabel("epoch")
            plt.show()

    def plotResults(self):
        if self.trainingType == "Not trained yet":
            print(self.trainingType)
        else:
            plt.plot(
                range(len(self.fittedValues)),
                self.fittedValues.cpu().numpy(),
                "r",
                label="trained",
            )
            plt.plot(range(len(self.Y)), self.Y.cpu().numpy(), "b", label="original")
            plt.legend(loc="upper left")
            plt.show()

    def predict(self, inputVar):
        inputVar = inputVar.clone().detach().to(self.device).float()
        [layerFour, wSum, w] = self.forwardHalfPass(inputVar)

        layerFive = layerFour @ self.consequents

        return layerFive

    def forwardHalfPass(self, Xs):
        layerFour = []
        wSum = []

        for pattern in range(len(Xs[:, 0])):
            layerOne = self.memClass.evaluateMF(Xs[pattern, :].cpu().numpy())

            miAlloc = [
                [layerOne[x][self.rules[row, x]] for x in range(len(self.rules[0]))]
                for row in range(len(self.rules))
            ]
            layerTwo = torch.tensor(
                [np.prod(x) for x in miAlloc], device=self.device, dtype=torch.float32
            )
            if pattern == 0:
                w = layerTwo
            else:
                w = torch.vstack((w, layerTwo))

            wSum.append(torch.sum(layerTwo))
            if pattern == 0:
                wNormalized = layerTwo / wSum[pattern]
            else:
                wNormalized = torch.vstack((wNormalized, layerTwo / wSum[pattern]))

            layerThree = layerTwo / wSum[pattern]
            rowHolder = torch.cat(
                [
                    x
                    * torch.cat(
                        (Xs[pattern, :], torch.tensor([1.0], device=self.device))
                    )
                    for x in layerThree
                ]
            )
            layerFour.append(rowHolder)

        w = w.T
        wNormalized = wNormalized.T

        layerFour = torch.stack(layerFour)

        return layerFour, wSum, w

    def backprop(self, columnX, columns, theWSum, theW, theLayerFive):
        paramGrp = [0] * len(self.memFuncs[columnX])
        for MF in range(len(self.memFuncs[columnX])):
            parameters = torch.empty(
                len(self.memFuncs[columnX][MF][1]), device=self.device
            )
            timesThru = 0
            for alpha in sorted(self.memFuncs[columnX][MF][1].keys()):
                bucket3 = torch.empty(len(self.X), device=self.device)
                for rowX in range(len(self.X)):
                    varToTest = self.X[rowX, columnX]
                    tmpRow = torch.empty(len(self.memFuncs), device=self.device)
                    tmpRow.fill_(varToTest)

                    bucket2 = torch.empty(self.Y.ndim, device=self.device)
                    for colY in range(self.Y.ndim):
                        rulesWithAlpha = torch.tensor(
                            np.where(self.rules[:, columnX].cpu().numpy() == MF)[0],
                            device=self.device,
                        )
                        adjCols = np.delete(columns, columnX)
                        mf_name = self.memFuncs[columnX][MF][0]
                        mf_parameters = self.memFuncs[columnX][MF][1]
                        senSit = partial_dmf(
                            self.X[rowX, columnX].cpu().numpy(),
                            mf_name,
                            mf_parameters,
                            alpha,
                        )

                        dW_dAplha = senSit * torch.tensor(
                            [
                                np.prod(
                                    [
                                        self.memClass.evaluateMF(tmpRow.cpu().numpy())[
                                            c
                                        ][self.rules[r, c]]
                                        for c in adjCols
                                    ]
                                )
                                for r in rulesWithAlpha
                            ],
                            device=self.device,
                        )

                        bucket1 = torch.empty(len(self.rules[:, 0]), device=self.device)
                        for consequent in range(len(self.rules[:, 0])):
                            fConsequent = torch.dot(
                                torch.cat(
                                    (
                                        self.X[rowX, :],
                                        torch.tensor([1.0], device=self.device),
                                    )
                                ),
                                self.consequents[
                                    ((self.X.shape[1] + 1) * consequent) : (
                                        ((self.X.shape[1] + 1) * consequent)
                                        + (self.X.shape[1] + 1)
                                    ),
                                    colY,
                                ],
                            )
                            acum = 0
                            if consequent in rulesWithAlpha:
                                acum = (
                                    dW_dAplha[rulesWithAlpha == consequent]
                                    * theWSum[rowX]
                                )

                            acum = acum - theW[consequent, rowX] * torch.sum(dW_dAplha)
                            bucket1[consequent] = fConsequent * (
                                acum / (theWSum[rowX] ** 2)
                            )

                        sum1 = torch.sum(bucket1)

                        if self.Y.ndim == 1:
                            bucket2[colY] = (
                                sum1 * (self.Y[rowX] - theLayerFive[rowX, colY]) * (-2)
                            )
                        else:
                            bucket2[colY] = (
                                sum1
                                * (self.Y[rowX, colY] - theLayerFive[rowX, colY])
                                * (-2)
                            )

                    sum2 = torch.sum(bucket2)
                    bucket3[rowX] = sum2

                sum3 = torch.sum(bucket3)
                parameters[timesThru] = sum3
                timesThru += 1

            paramGrp[MF] = parameters

        return paramGrp
