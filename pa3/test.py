from __future__ import annotations
import numpy as np
import os
import sys
from scipy.optimize import check_grad
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.ann.activations.tanh import Tanh
from src.ann.clf import ANNC
from src.ann.data.header import ContinuousFeature, GroundTruth, Header
from src.ann.losses.cce import CategoricalCrossEntropy
from src.ann.optimizers.sgd import SGDOptimizer


def check_cce_grad() -> None:
    cce = CategoricalCrossEntropy()
    Y_gt = np.array([[1., 0., 0.], [0., 1., 0.]])
    Y_hat = np.array([ [0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    def f(x):
        return cce.forward(x.reshape(Y_hat.shape), Y_gt)

    def grad(x):
        return cce.backward(x.reshape(Y_hat.shape), Y_gt).ravel()

    error = check_grad(f, grad, Y_hat.ravel())
    print(f"Gradient check error: {error}")


def test_iris() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345, stratify=y)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    X_train = (X_train - mu)/sd
    X_test = (X_test - mu)/sd
    header = Header()
    for i in range(X.shape[1]):
        header.add_feature(ContinuousFeature("x%d" % i, i))
    header.add_ground_truth(GroundTruth([str(c) for c in np.unique(y)]))
    np.random.seed(12345)
    model = ANNC(header, 10, Tanh())
    optimizer = SGDOptimizer(model.parameters(), 1e-1)
    model.fit(X_train, y_train, optimizer, gamma=0.0, epsilon=1e-8, max_iters=3000)
    acc = np.mean(model.predict(X_test) == y_test)
    print(f"iris test accuracy {acc}")

### cd pa3 && /opt/anaconda3/bin/python test.py
if __name__ == "__main__":
    check_cce_grad()
