"""
Utitlies for creating spiral datasets.
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


class ArchimedeanSpiral2D(DenseDesignMatrix):
    """
    Generates data for a 2D origin centered Archimedean spiral (i.e. a spiral
    of the form

       r = a*\theta + b

    for any real numbers a and b in polar coordinates).
    """
    _default_seed = 1

    def __init__(self, turn_rate=1, distance=0,
                 bounds_radius=2*math.pi, samples=1001,
                 positive_sample_rate=0.5, rng=None, epsilon_multiplier=2):
        """
        Parameters
        ----------
        turn_rate : double
            Amount by which to turn the spiral.
        distance: double
            Distance between successive turns.
        bounds_radius : double
            Radius of bounding circle. This is meant to keep the size
            of the data set tractable.
        samples : int
            Number of samples to generate.
        positive_sample_rate : float
            Real value in range [0,1] identifying the percentage of samples
            that should be positive.
        epsilon_multiplier : int
            Scales the sensitivity of comparison between numbers.
        rng : int
            Seed for random number generator.
        """

        # Validate parameters and set members values
        self.turn_rate = turn_rate
        self.distance = distance

        assert(bounds_radius > 0)  # What does a negative radius mean?
        assert(bounds_radius > distance)
        self.bounds_radius = bounds_radius
        self.max_theta = (self.bounds_radius - self.distance)/self.turn_rate

        assert(samples > 0)
        self.samples = samples

        assert(positive_sample_rate >= 0)
        assert(positive_sample_rate <= 1)
        self.positive_sample_rate = positive_sample_rate

        assert(epsilon_multiplier > 1)
        self.epsilon = epsilon_multiplier*sys.float_info.epsilon

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        (X, y) = self._generate_data()

        super(ArchimedeanSpiral2D, self).__init__(X=X, y=y)

    def _generate_data(self):
        """
        Generates X and y matrices for DenseDesignMatrix initialization
        function.
        """

        # Create random samples
        X = numpy.random.rand(self.samples, 2)
        X[:, 0] = self.bounds_radius*X[:, 0]
        X[:, 1] = self.max_theta*X[:, 1]
        y = numpy.zeros(self.samples)

        def onSpiral(samp):
            r = (self.turn_rate*samp[1] + self.distance)
            return abs(samp[0] - r) < self.epsilon

        # Label positive points (assumed all negative initially)
        for i in range(X.shape[0]):
            if onSpiral(X[i, ...]):
                y[i] = 1

        # Ensure we have the requested number of samples
        pos_samps = int(y.sum())
        needed_pos_samps = self.samples * self.positive_sample_rate
        needed_pos_samps = int(math.ceil(needed_pos_samps))

        def getRandIdxs(count, label):
            idxs = numpy.where(y == label)[0]
            numpy.random.shuffle(idxs)
            return idxs

        # Case: too many examples
        # Probabilistically speaking this will not happen.
        if pos_samps > needed_pos_samps:
            pos_samps_diff = pos_samps - needed_pos_samps
            idxs = getRandIdxs(pos_samps_diff, 1)

            for i in range(pos_samps_diff):
                idx = idxs[i]
                y[idx] = 0
                while onSpiral(X[idx, ...]):
                    X[idx, 0] = self.bounds_radius*rng.Random.rand()

        # Case: too few examples
        elif pos_samps < needed_pos_samps:
            pos_samps_diff = needed_pos_samps - pos_samps
            idxs = getRandIdxs(pos_samps_diff, 0)

            for i in range(pos_samps_diff):
                idx = idxs[i]
                X[idx, 0] = self.turn_rate*X[idx, 1] + self.distance
                y[idx] = 1

        return (X, y)


def test_generate_data():
    s = ArchimedeanSpiral2D(positive_sample_rate=1, bounds_radius=2*math.pi)

    (X, y) = s._generate_data()

    for i in range(X.shape[0]):
        print str(X[i, ...]) + ' ' + str(y[i])

    pos_samps = y.sum()
    expected_pos_samps = math.ceil(s.samples * s.positive_sample_rate)

    print "Samples: " + str(s.samples)
    print "Positive sample rate: " + str(s.positive_sample_rate)
    print "Expected positive samples: " + str(expected_pos_samps)
    print "Positive samples: " + str(pos_samps)

    assert(expected_pos_samps == pos_samps)

    Z = numpy.zeros(X.shape)
    for i in range(X.shape[0]):
        Z[i, 0] = X[i, 0]*math.cos(X[i, 1])
        Z[i, 1] = X[i, 0]*math.sin(X[i, 1])

    plt.scatter(Z[:, 0], Z[:, 1])
    plt.show()

    #Z = numpy.random.rand(s.samples, 2)
    #plt.scatter(Z[:, 0], Z[:, 1])
    #plt.show()

if __name__ == "__main__":
    test_generate_data()