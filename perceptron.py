# This file implements the perceptron classifier

import numpy as np


class Perceptron(object):
    """
    Parameters
    --------------------
    eta - learning rate
    n_epochs - passes over the training set

    Attributes
    --------------------
    w_ - weights after fitting
    errors_ - number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_epochs=10):
        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """
        Parameters
        --------------------
        X - data
        y - labels

        Returns
        --------------------
        self
        """

        self.w_ = np.zeros(range(1 + X.shape[1]))
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ Calculate net input function """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# TODO: extend class for multi-class classification, using One-vs.-All
