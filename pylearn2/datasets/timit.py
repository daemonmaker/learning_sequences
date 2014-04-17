"""
Utitlies for accessing TIMIT dataset.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import os
import sys
import numpy
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import pdb


class TIMITPerPhone(DenseDesignMatrix):
    """
    Loads specified dataset created from the TIMIT dataset by Laurent Dinh
    into a matrix for time series prediction and generation.
    """
    _default_seed = 1
    _data_dir = '/data/lisa/data/timit/readable/per_phone'

    def __init__(
        self, phone, frame_length, target_width=1,
        max_examples=None, random_examples=False,
        rng=None
    ):
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
        self.data = data
        idxs = numpy.arange(len(data))

        if self.random_examples:
            numpy.random.shuffle(idxs)

        if self.max_examples is not None:
            idxs = idxs[:self.max_examples]

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
    t = TIMITPerPhone('aa', 240, max_examples=2, random_examples=True)

    pdb.set_trace()

    # Plot some random samples
    plt.plot(t.X[0])
    plt.show()

if __name__ == "__main__":
    testload_data()
