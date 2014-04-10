"""
Utilities for analyzing spiral experiments.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import sys
import getopt
import pdb
import theano
import theano.tensor as T
import numpy
import matplotlib.pyplot as plt
import cPickle


def main():
    model_path = sys.argv[1:]
    if len(model_path) != 1:
        print "Usage: analyze_henonmap.py [path/to/model.pkl]"
        return 1
    else:
        model_path = model_path[0]

    model = cPickle.load(open(model_path, 'rb'))

    v = T.matrix("v")
    f = theano.function([v], [model.fprop(v)])

    # Determine whether the model is classification or regression
    # Number of output dims is one for binary classification and
    # two for regression.
    if (model.layers[1].dim == 1):
        right = -10
        left = 10
        dist = 0.125
        t = numpy.arange(right, left, dist)
        x, y = numpy.meshgrid(t, t)
        z = numpy.asarray(zip(x.flatten(), y.flatten()))

        #pdb.set_trace()

        points_per_side = (left - right)/dist
        #results = numpy.zeros((points_per_side, points_per_side))
        results = numpy.zeros((z.shape[0], 1))
        for i in range(z.shape[0]):
            input = z[i, :]
            input = input.reshape((input.shape[0], 1))
            output = f(input)
            #print output[0].flatten()
            results[i] = output[0].flatten()

        im = results.reshape((points_per_side, points_per_side))

        plt.imshow(im)
        plt.gca().invert_yaxis()
        plt.show()

    else:
        results = numpy.zeros((1000, 2))
        for i in range(1, results.shape[0]):
            input = results[i-1, :]
            input = input.reshape((input.shape[0], 1))
            output = f(input)
            results[i, :] = output[0].flatten()

        centers = model.layers[0].centers
        plt.scatter(centers[:, 0], centers[:, 1], color='red')

        plt.scatter(results[:, 0], results[:, 1])

        plt.show()


if __name__ == '__main__':
    main()
