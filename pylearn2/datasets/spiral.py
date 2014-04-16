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
import cPickle
import pdb


class ArchimedeanSpiral2D(DenseDesignMatrix):
    """
    Generates data for a 2D origin centered Archimedean spiral (i.e. a spiral
    of the form

       r = a*\theta + b

    for any real numbers a and b in polar coordinates).
    """
    _default_seed = 1

    def __init__(
        self, space='r,theta', turn_rate=1, distance=0,
        bounds_radius=2*math.pi, samples=1000,
        positive_sample_rate=0.5, rng=None, epsilon_multiplier=2,
        task='classification', load_path=None, save_path=None
    ):
        """
        Parameters
        ----------
        space : string
            Whether to generate the data as radius-angle pairs ('r,theta') or
            x-y pairs ('x,y').
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
            that should be positive. Only applicable for classification tasks.
        epsilon_multiplier : int
            Scales the sensitivity of comparison between numbers.
        rng : int
            Seed for random number generator.
        task : string
            Whether targets should be binary values for 'classification' or the
            next value in the sequence for reg'regression'.
        load_path : string
            Path from which to load data.
        save_path : string
            Path to which the data should be saved.
        """

        # Validate parameters and set member variables
        assert(space == 'x,y' or space == 'r,theta')
        self.space = space

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

        assert(task == 'classification' or task == 'regression')
        self.task = task

        if (task == 'regression'):
            self.positive_sample_rate = 1

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        if load_path is None:
            (X, y) = self._generate_data()
        else:
            (X, y) = cPickle.load(open(load_path, 'rb'))

        if save_path is not None:
            cPickle.dump((X, y), open(save_path, 'wb'))

        super(ArchimedeanSpiral2D, self).__init__(X=X, y=y)

    def _generate_data(self):
        """
        Generates X and y matrices for DenseDesignMatrix initialization
        function.
        """

        #pdb.set_trace()

        # Create random samples
        y = numpy.zeros((self.samples, 1))

        # Regression case
        if self.task == 'regression':
            samps = self.samples + 1

            deltaT = self.max_theta / float(self.samples)
            X = numpy.zeros((samps, 2))
            for i in range(1, X.shape[0]):
                X[i, 1] = X[i-1, 1] + deltaT
                X[i, 0] = self.turn_rate*X[i-1, 1] + self.distance

            y = X[1:, :]  # First example is not a target
            X = X[:-1, :]  # Last example is only a target

        # Classification case
        else:
            X = numpy.random.rand(self.samples, 2)
            X[:, 0] = self.bounds_radius*X[:, 0]
            X[:, 1] = self.max_theta*X[:, 1]

            def onSpiral(samp):
                r = (self.turn_rate*samp[1] + self.distance)
                return abs(samp[0] - r) < self.epsilon

            # Label positive points (assumed all negative initially)
            for i in range(X.shape[0]):
                if onSpiral(X[i, ...]):
                    y[i, 0] = 1

            # Ensure we have the requested number of samples
            pos_samps = int(y.sum())
            needed_pos_samps = self.samples * self.positive_sample_rate
            needed_pos_samps = int(math.ceil(needed_pos_samps))

            def getRandIdxs(count, label):
                idxs = numpy.copy(numpy.where(y == label)[0])
                numpy.random.shuffle(idxs)
                return idxs

            # Case: too many examples
            # Probabilistically speaking this will not happen.
            if pos_samps > needed_pos_samps:
                pos_samps_diff = pos_samps - needed_pos_samps
                idxs = getRandIdxs(pos_samps_diff, 1)

                for i in range(pos_samps_diff):
                    idx = idxs[i]
                    y[idx, 0] = 0
                    while onSpiral(X[idx, ...]):
                        X[idx, 0] = self.bounds_radius*rng.Random.rand()

            # Case: too few examples
            elif pos_samps < needed_pos_samps:
                pos_samps_diff = needed_pos_samps - pos_samps
                idxs = getRandIdxs(pos_samps_diff, 0)

                for i in range(pos_samps_diff):
                    idx = idxs[i]
                    X[idx, 0] = self.turn_rate*X[idx, 1] + self.distance
                    y[idx, 0] = 1

        # Convert to x,y space if appropriate
        if self.space == 'x,y':
            for i in range(X.shape[0]):
                x_temp = X[i, 0]*math.cos(X[i, 1])
                y_temp = X[i, 0]*math.sin(X[i, 1])
                X[i, 0] = x_temp
                X[i, 1] = y_temp

        return (X, y)


def test_generate_data():
    convert_to_xy = False

    s = ArchimedeanSpiral2D(
        samples=1000,
        space='r,theta',
        task='classification',
    )

    (X, y) = s._generate_data()

    for i in range(X.shape[0]):
        print str(X[i, ...]) + ' ' + str(y[i])

    pos_samps = y.sum()
    expected_pos_samps = math.ceil(s.samples * s.positive_sample_rate)

    print "Task: " + s.task
    print "Space: " + s.space
    print "Samples: " + str(s.samples)
    print "Positive sample rate: " + str(s.positive_sample_rate)
    print "Expected positive samples: " + str(expected_pos_samps)

    if (s.task == 'classification'):
        print "Positive samples: " + str(pos_samps)

        assert(expected_pos_samps == pos_samps)

    if (convert_to_xy is True) and (s.space == 'r,theta'):
        Z = numpy.zeros(X.shape)
        for i in range(X.shape[0]):
            Z[i, 0] = X[i, 0]*math.cos(X[i, 1])
            Z[i, 1] = X[i, 0]*math.sin(X[i, 1])
    else:
        Z = X

    figure = plt.figure(facecolor="white")
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.title("2D Spiral")
    if (convert_to_xy is True):
        plt.ylabel('y')
        plt.xlabel('x')

    else:
        plt.ylabel('theta')
        plt.xlabel('r')

    #plt.show()

    fig_name = "trainging_data-"
    if convert_to_xy is True:
        fig_name += "x-y"
    else:
        fig_name += "r-theta"

    plt.savefig(fig_name)

    #Z = numpy.random.rand(s.samples, 2)
    #plt.scatter(Z[:, 0], Z[:, 1])
    #plt.show()

if __name__ == "__main__":
    test_generate_data()
