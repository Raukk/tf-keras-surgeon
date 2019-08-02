"""Identify which channels to delete."""

import numpy as np
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.models import Model

from tfkerassurgeon import utils


class ApozIdentifier:
    """
    This class uses APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero, 
    in a layer outputs to identify candidates for pruning, and return a list of votes, that score each Output based on prune priority

    Arguments:
        modifier (Optional): applied to the output, it is a power (allowing better distinction between items that are 95% zeros and 75% zeros)
            A larger number will have the effect that the vote magnitude will drop off rapidly as the % of zeros decreases (Defaults to 1.0)
    """
    def __init__(self, modifier = 1.0):
        # Set the static values
        self.modifier = modifier

    def get_votes(self, model, layer, data_generator, num_of_batches = 10):
        """Identify neurons with high Average Percentage of Zeros (APoZ).

        The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
        is a metric for the usefulness of a channel defined in this paper:
        "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient
        Deep Architectures" - [Hu et al. (2016)][]
        `high_apoz()` enables the pruning methodology described in this paper to be
        replicated.

        If node_indices are not specified and the layer is shared within the model
        the APoZ will be calculated over all instances of the shared layer.

        Args:
            model: A Keras model.
            layer: The layer whose channels will be evaluated for pruning.
            data_generator: The input value generator for testing with. This will be used to calculate
                the activations of the layer of interest.
            num_of_batches: defines how many data batches from the generator to use while calculating. (larger number is slower, but more accurate).

        Returns:
            List of the APoZ values for each channel in the layer.
        """

        if isinstance(layer, str):
            layer = model.get_layer(name=layer)

        # Check that layer is in the model
        if layer not in model.layers:
            raise ValueError('layer is not a valid Layer in model.')

        # all of the layer's inbound nodes which are in model are selected.
        node_indices = utils.find_nodes_in_model(model, layer)

        # Check for duplicate node indices
        if len(node_indices) != len(set(node_indices)):
            raise ValueError('`node_indices` contains duplicate values.')
        # Check that all of the selected nodes are in the layer
        elif not set(node_indices).issubset(layer_node_indices):
            raise ValueError('One or more nodes specified by `layer` and '
                             '`node_indices` are not in `model`.')

        data_format = getattr(layer, 'data_format', 'channels_last')

        # Perform the forward pass and get the activations of the layer.
        mean_calculator = utils.MeanCalculator(sum_axis=0)
        for node_index in node_indices:

            # build a new model that takes the input and gives the ouptput of the specific layer
            act_layer, act_index = utils.find_activation_layer(layer, node_index)
            output = act_layer.get_output_at(act_index)
            temp_model = Model(model.inputs, output)

            # run through num_of_batches times, 1 batch at a time. This is to keep the memory ussage down
            for i in range(0, num_of_batches):

                # generate teh activations
                a = temp_model.predict_generator(data_generator, steps = 1)

                # Ensure that the channels axis is last
                if data_format == 'channels_first':
                    a = np.swapaxes(a, 1, -1)
                # Flatten all except channels axis
                activations = np.reshape(a, [-1, a.shape[-1]])
                zeros = (activations == 0).astype(int)
                mean_calculator.add(zeros)

        # return the average as the vote, to the power of the modifier (default is 1.0)
        return mean_calculator.calculate() ** self.modifier

