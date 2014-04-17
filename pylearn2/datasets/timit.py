"""
Utitlies for accessing TIMIT dataset.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Junyoung Chung"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import os
import sys
import numpy
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.utils import serial
import matplotlib.pyplot as plt
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
import ipdb


class TIMITPerPhone(DenseDesignMatrix):
    """
    Loads specified dataset created from the TIMIT dataset by Laurent Dinh
    into a matrix for time series prediction and generation.
    """
    _default_seed = 1
    _data_dir = '/data/lisa/data/timit/readable/per_phone'

    def __init__(self,
                 phone,
                 frame_length,
                 target_width=1,
                 max_examples=None,
                 example_list=None,
                 random_examples=False,
                 test=False,
                 unit_norm=False,
                 standardize=False,
                 mean=None,
                 std=None,
                 rng=None):
        """
        Parameters
        ----------
        phone : string
            The phone to be loaded.
        max_examples : int
            The maximum number of examples to load.
        example_list : list
            Specify examples to generate.
        random_examples: boolean
            Whether to select the examples from the data set at random if they
            are not all to be used (e.g. when max_examples is less than the
            total number examples).
        test : bool
            Whether to use training or test set. False means train and True
            means train.
        unit_norm : bool
            Normalize individual signal with it's L2 norm.
        standardize : bool
            Normalize all examples.
        mean : float
            Mean of training set. Only used if standarize flag is on.
        std : float
            Standard deviation of training set. Only used if standarize flag is
            on.
        rng : int
            Seed for random number generator.
        """
        #ipdb.set_trace()
        # Validate parameters and set member variables
        file = 'wav_' + phone + '.npy'
        files = os.listdir(self._data_dir)
        assert(file in files)
        self.phone = phone
        self.file = file
        self.example_list = example_list

        if example_list is not None:
            assert(numpy.asarray(example_list).mean() >= 0)
            assert isinstance(example_list, list) is True

        # Flags for tr/te set and normalization
        assert(type(test) is bool)
        self.test = test

        assert(type(unit_norm) is bool)
        self.unit_norm = unit_norm

        assert(type(standardize) is bool)
        self.standardize = standardize

        if (self.test and self.standardize):
            assert(mean is not None and std is not None)

        self._mean = mean

        if std is not None:
            assert(std > 0)
        self._std = std

        self._mean_norm = 0

        assert(frame_length > 0)
        self.frame_length = frame_length

        assert(target_width > 0)
        self.target_width = target_width

        self.max_examples = None
        if (max_examples is not None):
            assert(max_examples > 0)
            self.max_examples = max_examples

        assert(type(random_examples) == bool)
        self.random_examples = random_examples

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        (X, y) = self._load_data()

        super(TIMITPerPhone, self).__init__(X=X, y=y)

    def _load_data(self):
        data = serial.load(os.path.join(self._data_dir, self.file))

        if self.example_list is not None:
            idxs = self.example_list
        else:
            if self.test is True:
                data = data[-500:]
            else:
                data = data[:-500]
            idxs = numpy.arange(len(data))

            if self.random_examples:
                numpy.random.shuffle(idxs)

            if self.max_examples is not None:
                idxs = idxs[:self.max_examples]

        data = data[idxs]
        # TODO - Remove this
        self.data = data

        if self.unit_norm is True:
            for i in range(data.shape[0]):
                exp_euclidean_norm = numpy.sqrt(numpy.square(data[i]).sum())
                data[i] /= exp_euclidean_norm
                self._mean_norm += exp_euclidean_norm
            self._mean_norm /= data.shape[0]

        if self.standardize is True:
            if self._mean is None or self._std is None:
                exp_sum = 0
                exp_var = 0
                exp_cnt = 0
                for i in range(data.shape[0]):
                    exp_sum += data[i].sum()
                    exp_var += (numpy.square(data[i])).sum()
                    exp_cnt += len(data[i])
                self._mean = exp_sum / exp_cnt
                exp_var = exp_var/exp_cnt - self._mean**2
                self._std = numpy.sqrt(exp_var)

            for i in range(data.shape[0]):
                data[i] = (data[i] - self._mean) / self._std

        # Do math to determine how many samples there will be and make space
        total_rows = 0
        record_len = self.frame_length + self.target_width
        for i in range(len(data)):
            total_rows += len(data[i]) - record_len

        X = numpy.zeros((total_rows, self.frame_length))
        y = numpy.zeros((total_rows, self.target_width))

        count = 0
        for i in range(len(data)):
            current_phone = data[i]
            current_phone_len = len(current_phone)
            for j in range(current_phone_len - record_len):
                frame_end = j + self.frame_length
                target_end = frame_end + self.target_width
                X[count, :] = current_phone[j:frame_end]
                y[count, :] = current_phone[frame_end:target_end]
                count += 1

        return (X, y)


def testload_data():
    """
    Routine for testing the loading of TIMIT phones
    """
    t = TIMITPerPhone(phone='aa',
                      frame_length=240,
                      max_examples=200,
                      random_examples=False,
                      unit_norm=True,
                      standardize=True)
    # Plot some random samples
    plt.plot(t.X[0])
    plt.show()

    return t

if __name__ == "__main__":
    data_specs = (CompositeSpace([VectorSpace(dim=240),
                                  VectorSpace(dim=1)]),
                  ('features', 'targets'))
    t = testload_data()
    it = t.iterator(mode='sequential', data_specs=data_specs,
                    num_batches=None, batch_size=100)
    X, y = it.next()
    ipdb.set_trace()
