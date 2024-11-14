<div>
  <img alt="CodeQL Analysis" src="https://github.com/lazuardy-tech/anfis/actions/workflows/github-code-scanning/codeql/badge.svg" />
  <img alt="Build" src="https://github.com/lazuardy-tech/anfis/actions/workflows/build.yml/badge.svg" />
  <img alt="PyPI License" src="https://img.shields.io/pypi/l/lazuardy-anfis" />
  <img alt="PyPI Version" src="https://img.shields.io/pypi/v/lazuardy-anfis" />
  <img alt="PyPI Python Version" src="https://img.shields.io/pypi/pyversions/lazuardy-anfis" />
  <img alt="PyPI Implementation" src="https://img.shields.io/pypi/implementation/lazuardy-anfis" />
</div>

## ∑ anfis

Adaptive Neuro Fuzzy Inference System Implementation in Python.

This project is a fork of [twmeggs/anfis](https://github.com/twmeggs/anfis), with bug fixes, optimizations, and improvements so that the package can be used in further projects. All credits goes to [twmeggs](https://github.com/twmeggs) for the original implementation.

<br/>

### ℹ️ About

This ANFIS package is essentially a Python refactoring of the R code created by the team a the BioScience Data Mining Group.

As an example of an ANFIS system, this Python code works (install and run the tests.py script to see it fit the some test data) but there is much left to do in order to improve the project. Documentation and doc strings still need a large amounts of work.

All useful contributions to make this a better project will be happily received.

<br/>

### 🚀 Getting Started

This package may then be installed by running:

```bash
pip install lazuardy_anfis
```

<br/>

### 🧪 Testing

Install anfis and navigate to the location of `src/tests.py`.

From the command line run:

```bash
python src/tests.py
```

Alternatively, from the same location launch `ipython` and run:

```bash
run src/tests.py
```

This will set up and fit an ANFIS model based on the data contained in `training_set.txt`, using 10 epochs. Plots of the fitting errors and the model predicted output are graphed.

<br/>

### ⭐ Features

Currently the implementation will support the use of three types of membership function:

- `gaussmf` : Gaussian
- `gbellmf` : Generalized bell
- `sigmf` : Sigmoid

This naming is taken from [scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy), a fuzzy logic toolkit for SciPy.

Each input variable can have an arbitrary number and mix of these membership functions.

A user can define the number of epochs that will be run. The returned ANFIS object can plot training errors, fitted results and the current shape of its membership functions (pre or post training).

<br/>

> This project is licensed under [MIT License](https://github.com/lazuardy-tech/anfis/blob/main/LICENSE).

> © [Lazuardy](https://lazuardy.tech) 2024. All rights reserved. <br/>
> PT Inovasi Kolektif Digital. <br/> [Terms of Service](https://lazuardy.tech/terms) | [Privacy Policy](https://lazuardy.tech/privacy)
