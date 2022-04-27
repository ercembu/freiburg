import numpy as np
from typing import Tuple


class BLR:
    """
        Class implementing Bayesian Linear Regression as described in the lecture slides
        __init__ for the class takes prior mean, prior variance, noise and bias (for DNGO)
        as input.

        _mu_pre is the prior mean, _sigma_pre is the prior covariance,
        _noise is the noise variance sigma_n^2 in the input data
    """

    def __init__(self, mu_pre: np.array, sigma_pre: np.array, noise: float, bias: bool = False):
        if np.count_nonzero(mu_pre) > 0:
            raise NotImplementedError("BLR currently only supports mu_pre=0")
        self._mu_pre = mu_pre
        self._sigma_pre = sigma_pre
        self._noise = noise
        self.bias = bias
        self._mu_post = None
        self._sigma_post = None

    def linreg_bayes(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        """ Compute the posterior distribution in this function which
            solves eqn. 2.8 from the GPML book.

        Args:
            X : Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample
            y : Numpy array of shape (n,) n - number of samples

        Returns:
            Posterior mean Numpy array of shape (D, ) D -  dimension of sample
            Posterior variance Numpy array of shape (D,D) D -  dimension of sample

        """
        if self.bias:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack([X, bias])

        # Using Dxn format used in the GPML book
        X = X.T

        # START TODO ########################
        # Implement eqn 2.8
        # Note that X was already transposed to be conform to the notation use in the book (examples ~ columns)
        # You are allowed to use np.linalg.inv()
        A = np.linalg.inv(self._sigma_pre) + np.dot(X, X.T) /self._noise
        Sigma_post = np.linalg.inv(A)

        mu_post = np.dot(Sigma_post, np.dot(X, y) /self._noise)
        # END TODO ########################
        self._mu_post = mu_post
        self._sigma_post = Sigma_post

        return mu_post, Sigma_post

    def posterior_predictive(self, X: np.array) -> Tuple[np.array, np.array]:
        """
            In this function you have to implement the posterior
            predictive distribution.
            Compute the mean and std. of each point x using eqn. 2.9.
        Args:
            X : Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample
            bias : if bias needed (DNGO case) add additional column to x
        Returns:
            predicted mean  Numpy array of shape (n, 1) n - number of samples and
            predicted std. Numpy array of shape (n, 1) n - number of samples and

        """

        # add additional bias column
        if self.bias:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack([X, bias])

        # Using Dxn format used in the GPML book
        X = X.T

        # START TODO ########################
        # Implement eqn 2.9
        # Note that X was already transposed to be conform to the notation use in the book (examples ~ columns)
        # Keep in mind that you have already computed some of the quantaties for eqn. 2.9.
        # To get the mean estimates, you only have to multiply the posterior mean estimate from eqn 2.8
        # (which were saved for the class in fit_blr_model()) with each of the x.
        # To get the std. estimate at each of the points in x, you pre- and post-multiply
        # the covariance estimate with x^T and x.
        x_mean = np.dot(X.T, self._mu_post)
        x_std = np.dot(np.dot(X.T, self._sigma_post), X)
        x_std = np.diagonal(x_std)
        x_mean = x_mean.reshape(-1, 1)
        x_std = x_std.reshape(-1, 1)
        # END TODO ########################

        return x_mean, x_std

    @property
    def Sigma_pre(self):
        """ getter for the prior covariance
        """
        return self._sigma_pre

    @property
    def Sigma_post(self):
        """ getter for the posterior covariance
        """
        return self._sigma_post

    @property
    def Mu_pre(self):
        """ getter for the prior mean
        """
        return self._mu_pre

    @property
    def Mu_post(self):
        """ getter for the posterior mean
        """
        return self._mu_post
