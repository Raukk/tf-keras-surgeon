"""Utilities used across other modules."""
import warnings
import numpy as np
import collections
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.activations import linear


def clean_copy(model):
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = model.__class__.from_config(model.get_config())
    new_model.set_weights(weights)
    return new_model


def get_channels_attr(layer):
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        channels_attr = 'units'
    elif 'filters' in layer_config.keys():
        channels_attr = 'filters'
    else:
        raise ValueError('This layer has not got any channels.')
    return channels_attr

def get_node_depth(model, node):
    """Get the depth of a node in a model.

    Arguments:
        model: Keras Model object
        node: Keras Node object

    Returns:
        The node depth as an integer. The model outputs are at depth 0.

    Raises:
        KeyError: if the node is not contained in the model.
    """
    for (depth, nodes_at_depth) in get_nodes_by_depth(model).items():
        if node in nodes_at_depth:
            return depth
    raise KeyError('The node is not contained in the model.')


def find_nodes_in_model(model, layer):
    """Find the indices of layer's inbound nodes which are in model"""
    model_nodes = get_model_nodes(model)
    node_indexes = [] # Renamed this since it was confusing with TF/Keras Node.node_indices
    for i, node in enumerate(get_inbound_nodes(layer)):
        if node in model_nodes:
            node_indexes.append(i)
    return node_indexes


def get_model_nodes(model):
    """Return all nodes in the model"""
    return [node for v in get_nodes_by_depth(model).values() for node in v]


def get_shallower_nodes(node):
    possible_nodes = get_outbound_nodes(node.outbound_layer)
    next_nodes = []
    for n in possible_nodes:
        for i, node_index in enumerate(item_to_list(n.node_indices)):
            if node == get_inbound_nodes(item_to_list(n.inbound_layers)[i])[node_index]:
                next_nodes.append(n)
    return next_nodes


def get_node_inbound_nodes(node):
    return [get_inbound_nodes(item_to_list(node.inbound_layers)[i])[node_index]
            for i, node_index in enumerate(item_to_list(node.node_indices))]


def get_inbound_nodes(layer):
    try:
        return getattr(layer, '_inbound_nodes')
    except AttributeError:
        warnings.warn("Please update keras to version 2.1.3 or greater."
                      "Support for earlier versions will be dropped in a "
                      "future release.")
        return layer.inbound_nodes


def get_outbound_nodes(layer):
    try:
        return getattr(layer, '_outbound_nodes')
    except AttributeError:
        warnings.warn("Please update keras to version 2.1.3 or greater."
                      "Support for earlier versions will be dropped in a "
                      "future release.")
        return layer.outbound_nodes


def get_nodes_by_depth(model):
    try:
        return getattr(model, '_nodes_by_depth')
    except AttributeError:
        warnings.warn("Please update keras to version 2.1.3 or greater."
                      "Support for earlier versions will be dropped in a "
                      "future release.")
        return model.nodes_by_depth


def get_node_index(node):
    for i, n in enumerate(get_inbound_nodes(node.outbound_layer)):
        if node == n:
            return i


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

            
def single_element(x):
    """If x contains a single element, return it; otherwise return x"""

    # If the item has a length, and the length is 1, return the object's item, otherwise just return the object
    if False == hasattr(x, '__len__') or len(x) == 1:
        x = x[0]

    # otherwise just return the item, since it's either not a list at all, or has multiple elements 
    return x


def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(
            np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


def item_to_list(item):
    """If the item is not a List or Array, this will make it a List with 1 item"""
    
    if item is None:
        return None

    # We don't need to do anything
    if type(item) is list:
        return item

    # We'll let it stay an Array stay since it should work the same
    if isinstance(item, np.ndarray):
        return item

    # otherwise return a list with just the one element
    return [item]


# Note: this method is not currently used. If it ends up not being needed, then remove it.
def item_to_np_array(item):
    """If the item is not a numpy Array, this will make it an Array, if it's not a list, then it will be a 1 lenght Array"""

    if item is None:
        return None

    # if it's already an Array, we're done
    if isinstance(item, np.ndarray):
        return item

    # if its a list, then convert to an Array
    if type(item) is list:
        return np.array(item)

    # Otherwise, make it a list with one item and then make that an Array
    return np.array([item])



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
