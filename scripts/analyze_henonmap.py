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

    alpha = 1.4
    beta = 0.3
    frame_length = 10
    samps = 10000

    v = T.matrix('v')
    f = theano.function([v], [model.fprop(v)])

    if model.layers[0].centers.shape[1] == 2:
        results = numpy.zeros((samps, 2))
        for i in range(1, results.shape[0]):
            input = results[i-1, :]
            input = input.reshape((input.shape[0], 1))
            output = f(input)
            #print output[0].flatten()
            results[i, :] = output[0].flatten()

        figure = plt.figure(facecolor="white")

        plt.title("Henon Map - One-Step Prediction")
        plt.scatter(results[:, 0], results[:, 1])
        plt.xlabel("x")
        plt.ylabel("y")
        #plt.show()
        plt.savefig("henonmap")

    else:
        X = numpy.zeros((samps, 2))
        for i in range(1, samps):
            X[i, 0] = 1 - alpha*X[i-1, 0]**2 + X[i-1, 1]
            X[i, 1] = beta*X[i-1, 0]

        prediction = numpy.zeros(X.shape)
        prediction[0:frame_length, :] = X[0:frame_length, :]
        for i in range(frame_length, samps):
            input = prediction[i-frame_length/2.0:i, :]
            input = input.flatten()
            input = input.reshape(1, input.shape[0])
            #pdb.set_trace()
            output = f(input)
            prediction[i, :] = output[0]

        generation = numpy.zeros(X.shape)
        #generation[0:frame_length, :] = X[0:frame_length, :]
        for i in range(frame_length, samps):
            input = generation[i-frame_length/2.0:i, :]
            input = input.flatten()
            input = input.reshape(1, input.shape[0])
            #pdb.set_trace()
            output = f(input)
            generation[i, :] = output[0]

        fig = plt.figure(facecolor="white", figsize=(12, 5), dpi=100)

        plt.subplot("131")
        plt.scatter(X[:, 0], X[:, 1])
        plt.title('Real System')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot("132")
        plt.scatter(prediction[:, 0], prediction[:, 1])
        plt.title('Prediction')
        plt.xlabel('x')

        plt.subplot("133")
        plt.scatter(generation[:, 0], generation[:, 1])
        plt.title('Generation')
        plt.xlabel('x')

        #plt.show()
        plt.savefig("henonmap")

        #pdb.set_trace()

if __name__ == '__main__':
    main()
