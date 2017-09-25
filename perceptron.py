# This file implements the perceptron classifier

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class Perceptron(object):
    """
    Perceptron classifier for binary
    classification problems.

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
        Perceptron classifier for binary
        classification problems.
        
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


class PerceptronOvA(object):
    """
    Perceptron classifier for multiclass problems,
    using the One-vs.-All method.
    
    Parameters
    --------------------
    eta - learning rate
    n_epochs - passes over the training set

    Attributes
    --------------------
    w_ - weights after fitting
    b_ - bias after fitting
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

        self.w_ = np.zeros(shape=(X.shape[1], len(y.unique())))
        self.b_ = np.zeros(len(y.unique()))
        lb = LabelBinarizer(neg_label=-1, pos_label=1)
        self.yi_ = lb.fit_transform(y)

        for _ in range(self.n_epochs):
            for xi, target in zip(X.as_matrix(), self.yi_):
                update = self.eta*(target - self.predict(xi))
                self.w_ += np.outer(xi,update)
                self.b_ += update
        return self

    def net_input(self, X):
        """ Calculate net input function """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def gen_csv(self, X):
        """ Generate a csv file with the ImageId and the Label """
        prediction = np.argmax(self.predict(X), axis=1)
        df_sub = pd.DataFrame({"ImageId": X.index.values+1, "Label": prediction})
        return df_sub










