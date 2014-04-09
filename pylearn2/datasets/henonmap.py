"""
Class for creating Henon map datasets.
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
import cPickle
import pdb
from research.code.scripts.segmentaxis import segment_axis


class HenonMap(DenseDesignMatrix):
    """
    Generates data for Henon map, i.e.

       x_{n+1} = 1 - \alpha*x_n^2 + y_n
       y_{n+1} = \beta*x_n
    """
    _default_seed = 1

    def __init__(
        self, alpha=1.4, beta=0.3, init_state=numpy.array([0, 0]),
        samples=1000, frame_length=10, rng=None,
        load_path=None, save_path=None
    ):
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
           Number of desired samples. Must be an integer multiple of
           frame_length.
        frame_length : int
           Number of samples contained in a frame. Must divide samples.
        rng : int
            Seed for random number generator.
        load_path : string
            Path from which to load data.
        save_path : string
            Path to which the data should be saved.
        load_path : string
            Path from which to load data.
        save_path : string
            Path to which the data should be saved.
        """

        # Validate parameters and set members values
        self.alpha = alpha
        self.beta = beta

        assert(samples % frame_length == 0)

        assert(frame_length > 0)
        self.frame_length = frame_length

        assert(samples > 0)
        self.samples = samples

        assert(init_state.shape in [(2,), (2)])
        self.init_state = init_state

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        if (load_path is None):
            (X, y) = self._generate_data()
        else:
            (X, y) = cPickle.load(open(load_path, 'rb'))

        if (save_path is not None):
            cPickle.dump((X, y), open(save_path, 'wb'))

        super(HenonMap, self).__init__(X=X, y=y)

    def _generate_data(self):
        """
        Generates X matrix for DenseDesignMatrix initialization
        function.
        """
        X = numpy.zeros((self.samples+1, 2))
        X[0, :] = self.init_state
        y = numpy.zeros(self.samples)

        for i in range(1, X.shape[0]):
            X[i, 0] = 1 - self.alpha*X[i-1, 0]**2 + X[i-1, 1]
            X[i, 1] = self.beta*X[i-1, 0]

        last_target = X[-1, :]
        X = X[:-1, :]
        X.reshape((1, self.samples*2))  # Flatten

        Z = segment_axis(X, length=self.frame_length*2, overlap=0)

        y = numpy.zeros((Z.shape[0], 2))
        y[:-1, :] = Z[1:, 0:2]
        y[-1, :] = last_target  # X[-1, :]

        return (Z, y)


def test_generate_data():
    """
    Routine for testing the henon map data generation.
    """
    h = HenonMap(samples=10000, frame_length=1)

    (X, y) = h._generate_data()

    for i in range(X.shape[0]):
        print str(X[i, ...]) + ' ' + str(y[i])

    print "Alpha: " + str(h.alpha)
    print "Beta: " + str(h.beta)
    print "Samples: " + str(h.samples)
    print "Frame length: " + str(h.frame_length)

    pdb.set_trace()

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

if __name__ == "__main__":
    test_generate_data()
