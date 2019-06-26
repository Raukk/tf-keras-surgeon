"""Utilities used in Core module."""
import warnings
import numpy as np
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.activations import linear



def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(
            np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


# Model Utils

def clean_copy(model):

    """Returns a copy of the model without other model uses of its layers."""

    # TODO? this is probably better than saving and reloading, but basically the same process

    # grab the Weights and hold them
    weights = model.get_weights()
    # grab the Config of the one to copy 
    old_model_config = model.get_config()
    # create a fresh Copy based on the old Config
    new_model = model.__class__.from_config(old_model_config)
    # Set the Weights based on the onld Model
    new_model.set_weights(weights)

    return new_model


# Layer Utils

def get_channels_attr(layer):

    """Gets the channels, either Units, or Filters"""

    layer_config_keys = layer.get_config().keys()

    if 'units' in layer_config_keys:
        return 'units'
    elif 'filters' in layer_config_keys:
        return 'filters'
    else:
        raise ValueError('This layer has not got any channels.')


# Node Utils

def get_node_inbound_nodes(node):

    return [get_inbound_nodes(node.inbound_layers[i])[node_index]

            for i, node_index in enumerate(node.node_indices)]


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

    # Is this really the bset way to get the depth? Really TF/Keras? No util that does this in the Framework?

    # Get every node and it's depth.
    nodes_by_depth = get_nodes_by_depth(model)
    node_items_by_depth = nodes_by_depth.items()

    # then look for our node, and return whatever depth it is
    for (depth, nodes_at_depth) in node_items_by_depth:
        if node in nodes_at_depth:
            return depth


    raise KeyError('The node is not contained in the model.')

