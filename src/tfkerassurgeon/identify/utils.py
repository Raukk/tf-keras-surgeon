"""Utilities used in Identify module."""
import warnings
import numpy as np
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.activations import linear


class MeanCalculator:
    def __init__(self, sum_axis):
        self.values = None
        self.n = 0
        self.sum_axis = sum_axis

    def add(self, v):
        if self.values is None:
            self.values = v.sum(axis=self.sum_axis)
        else:
            self.values += v.sum(axis=self.sum_axis)
        self.n += v.shape[self.sum_axis]

    def calculate(self):
        return self.values / self.n


def find_activation_layer(layer, node_index):
    """

    Args:
        layer(Layer):
        node_index:
    """
    output_shape = layer.get_output_shape_at(node_index)
    maybe_layer = layer
    node = get_inbound_nodes(maybe_layer)[node_index]
    # Loop will be broken by an error if an output layer is encountered
    while True:
        # If maybe_layer has a nonlinear activation function return it and its index
        activation = getattr(maybe_layer, 'activation', linear)
        if activation.__name__ != 'linear':
            if maybe_layer.get_output_shape_at(node_index) != output_shape:
                ValueError('The activation layer ({0}), does not have the same'
                           ' output shape as {1}'.format(maybe_layer.name,
                                                         layer.name))
            return maybe_layer, node_index

        # If not, move to the next layer in the datastream
        next_nodes = get_shallower_nodes(node)
        # test if node is a list of nodes with more than one item
        if len(next_nodes) > 1:
            ValueError('The model must not branch between the chosen layer'
                       ' and the activation layer.')
        node = next_nodes[0]
        node_index = get_node_index(node)
        maybe_layer = node.outbound_layer

        # Check if maybe_layer has weights, no activation layer has been found
        if maybe_layer.weights and (
                not maybe_layer.__class__.__name__.startswith('Global')):
            AttributeError('There is no nonlinear activation layer between {0}'
                           ' and {1}'.format(layer.name, maybe_layer.name))


# Node Utils

def get_shallower_nodes(node):

    possible_nodes = get_outbound_nodes(node.outbound_layer)

    next_nodes = []

    for n in possible_nodes:

        for i, node_index in enumerate(n.node_indices):

            if node == get_inbound_nodes(n.inbound_layers[i])[node_index]:

                next_nodes.append(n)

    return next_nodes


def get_node_index(node):

    # TODO: Is there really no built in that lets a node know what it's index is?

    for i, n in enumerate(get_inbound_nodes(node.outbound_layer)):

        if node == n:

            return i


# Layer Utils

def get_outbound_nodes(layer):

    try:

        return getattr(layer, '_outbound_nodes')

    except AttributeError:

        warnings.warn("Please update keras to version 2.1.3 or greater."
                      "Support for earlier versions will be dropped in a "
                      "future release.")

        return layer.outbound_nodes

