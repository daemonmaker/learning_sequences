"""
Utilities for creating Henon map datasets.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import sys
import numpy
import matplotlib.pyplot as plt
import math
from pylearn2.datasets import DenseDesignMatrix
import pdb


class HenonMap(DenseDesignMatrix):
    """
    Generates data for Henon map, i.e.

       x_{n+1} = 1 - \alpha*x_n^2 + y_n
       y_{n+1} = \beta*x_n
    """
    _default_seed = 1

    def __init__(self, alpha=1.4, beta=0.3, init_state=numpy.array([0, 0]),
                 samples=1000, rng=None):
        """
        Parameters
        ----------
        alpha : double
           Alpha parameter in equations above.
        beta : double
           Beta parameter in equations above.
        init_state : ndarray
           The initial state of the system of size 2.
        samples : int
           Number of desired samples.
        rng : int
            Seed for random number generator.
        """

        # Validate parameters and set members values
        self.alpha = alpha
        self.beta = beta

        assert(samples > 0)
        self.samples = samples

        assert(init_state.shape in [(2,), (2)])
        self.init_state = init_state

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        (X, y) = self._generate_data()

        super(HenonMap, self).__init__(X=X, y=y)

    def _generate_data(self):
        """
        Generates X and y matrices for DenseDesignMatrix initialization
        function.
        """
        X = numpy.zeros((self.samples, 2))
        X[0, :] = self.init_state
        y = numpy.ones(self.samples)

        for i in range(1, X.shape[0]):
            X[i, 0] = 1 - self.alpha*X[i-1, 0]**2 + X[i-1, 1]
            X[i, 1] = self.beta*X[i-1, 0]

        return (X, y)


def test_generate_data():
    """
    Routine for testing the henon map data generation.
    """
    h = HenonMap()

    (X, y) = h._generate_data()

    #for i in range(X.shape[0]):
    #    print str(X[i, ...]) + ' ' + str(y[i])

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

if __name__ == "__main__":
    test_generate_data()
