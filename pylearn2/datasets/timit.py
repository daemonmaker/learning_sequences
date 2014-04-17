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
                 random_examples=False,
                 test=False,
                 unit_norm=False,
                 normalize=False,
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
        random_examples: boolean
            Whether to select the examples from the data set at random if they
            are not all to be used (e.g. when max_examples is less than the
            total number examples).
        """

        # Validate parameters and set member variables
        file = 'wav_' + phone + '.npy'
        files = os.listdir(self._data_dir)
        assert(file in files)
        self.phone = phone
        self.file = file

        # Flags for tr/te set and normalization
        self.test = test
        self.unit_norm = unit_norm
        self.normalize = normalize
        self._mean = mean
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

        if self.test is True:
            data = data[-300:]
        else:
            data = data[:-300]

        # TODO - Remove this
        self.data = data

        idxs = numpy.arange(len(data))

        if self.random_examples:
            numpy.random.shuffle(idxs)

        if self.max_examples is not None:
            idxs = idxs[:self.max_examples]

        if self.unit_norm is True:
            for i, example in enumerate(data):
                exp_euclidean_norm = numpy.sqrt(numpy.square(example).sum())
                data[i] = example / exp_euclidean_norm
                self._mean_norm += exp_euclidean_norm
            self._mean_norm /= data.shape[0]

        if self.normalize is True:
            if self._mean is not None and self._std is not None:
                for i, example in enumerate(data):
                    data[i] = (example - self._mean) / self._std
            else:
                exp_sum = 0
                exp_cnt = 0
                for i, example in enumerate(data):
                    exp_sum += example.sum()
                    exp_cnt += len(example)
                exp_mean = exp_sum / exp_cnt
                exp_sqr_sum = 0
                for i, example in enumerate(data):
                    exp_sqr_sum += numpy.cast[float](numpy.square(example -
                                                     exp_mean).sum())
                exp_std = numpy.sqrt(exp_sqr_sum / exp_cnt)
                self._mean = exp_mean
                self._std = exp_std
                for i, example in enumerate(data):
                    data[i] = (example - self._mean) / self._std

        # Do math to determine how many samples there will be and make space
        total_rows = 0
        record_len = self.frame_length + self.target_width
        for i in idxs:
            total_rows += len(data[i]) - record_len

        X = numpy.zeros((total_rows, self.frame_length))
        y = numpy.zeros((total_rows, self.target_width))

        count = 0
        for i in idxs:
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
                      max_examples=100,
                      random_examples=False,
                      unit_norm=True,
                      normalize=True)
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
