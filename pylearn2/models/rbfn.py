"""
Radial Basis Function Network
"""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

from pylearn2.models


class RadialBasisFunction(Linear):
    """
    A layer that performs an affine transformation of its (vectorial)
    input followed by a (Gaussian) radial basis function.
    """

    def __init__(self, training_data,
                 standard_deviation=1, **kwargs):
        super(RadialBasisFunction, self).__init__(**kwargs)
        assert standard_deviation > 0
        self.standard_deviation = standard_deviation

        pdb.set_trace()

        # Select centers
        centers = np.zeros(
            self.dim,
            training_data.frame_length*training_data.frames_per_example)

        num_sequences = training_data.raw_wav.shape[0]
        samples_idx = np.arange(num_sequences)
        samples_idx = np.random.shuffle(samples_idx)

        for i in range(self.dim):
            id1 = samples_idx[i]
            id2 = np.random.randint(training_data.raw_wav[id1].shape[0])
            centers[id1] = training_data.raw_wav[id1][id2]

        self.centers = centers.dimshuffle('x', 0, 1)

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = ((state_below - self.centers) ** 2).sum(axis=1)
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        p = t.exp(-z/(2*self.standard_deviation**2))
        if self.layer_name is not None:
            p.name = self.layer_name + '_p_'

        return self.transformer.lmul(p)
