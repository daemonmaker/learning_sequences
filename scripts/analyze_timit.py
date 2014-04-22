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
import ipdb
import theano
import theano.tensor as T
import numpy
import matplotlib.pyplot as plt
import cPickle
from learning_sequences.pylearn2.datasets.timit import TIMITPerPhone


def main():
    model_path = sys.argv[1:]
    if len(model_path) != 1:
        print "Usage: analyze_henonmap.py [path/to/model.pkl]"
        return 1
    else:
        model_path = model_path[0]

    model = cPickle.load(open(model_path, 'rb'))

    v = T.matrix('v')
    f = theano.function([v], [model.fprop(v)])

    frame_length = 240
    t = TIMITPerPhone(
        'aa', frame_length, max_examples=1, random_examples=False,
        standardize=True, unit_norm=True,
        #test=True, mean=, std=,
    )

    #ipdb.set_trace()
    current_data = t.data[0]
    prediction_samps = current_data.shape[0]
    generation_samps = 1500

    plt.subplot("131")
    plt.plot(current_data)
    plt.title("Real signal")

    prediction = numpy.zeros((prediction_samps, 1))
    signal = current_data[0:frame_length]
    prediction[0:frame_length, :] = signal.reshape(frame_length, 1)
    for i in range(frame_length, prediction_samps):
        input = current_data[(i-frame_length):i]
        input = input.reshape(1, input.shape[0])
        output = f(input)
        prediction[i, :] = output[0]

    plt.subplot("132")
    plt.plot(prediction)
    plt.title('Prediction')

    #pdb.set_trace()

    generation = numpy.zeros((generation_samps, 1))
    signal = current_data[0:frame_length]
    generation[0:frame_length, :] = signal.reshape(frame_length, 1)
    for i in range(frame_length, generation_samps):
        input = generation[(i-frame_length):i, :]
        #pdb.set_trace()
        input = input.reshape(1, input.shape[0])
        output = f(input)
        generation[i, :] = output[0]

    plt.subplot("133")
    plt.plot(generation)
    plt.title('Generation')
    plt.savefig("analyze_timit")
    plt.show()
    #plt.savefig("timit.png")


if __name__ == '__main__':
    main()
