"""
Radial Basis Function Network
"""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

from pylearn2.models.mlp import *


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

        assert(training_data is not None)
        self.training_data = training_data

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = (state_below.flatten() - self.centers) ** 2
        z = z.sum(axis=1, keepdims=True).T
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        p = T.exp(-z/(2*self.standard_deviation**2))
        if self.layer_name is not None:
            p.name = self.layer_name + '_p_'

        return p

    def set_input_space(self, space):
        super(RadialBasisFunction, self).set_input_space(space)

        #pdb.set_trace()
        itr = self.training_data.iterator(
            mode='random_uniform',
            batch_size=self.dim,
            num_batches=1,  # We only want one set of points
        )
        self.centers = itr.next()

        del self.training_data

        #pdb.set_trace()
