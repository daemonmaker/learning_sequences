"""
Utilities for analyzing Henon map experiments.
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
    #pdb.set_trace()

    v = T.matrix('v')
    f = theano.function([v], [model.fprop(v)])

    results = numpy.zeros((10000, 2))
    for i in range(1, results.shape[0]):
        input = results[i-1, :]
        input = input.reshape((input.shape[0], 1))
        output = f(input)
        print output[0].flatten()
        results[i, :] = output[0].flatten()

    plt.scatter(results[:, 0], results[:, 1])
    plt.show()

if __name__ == '__main__':
    main()
