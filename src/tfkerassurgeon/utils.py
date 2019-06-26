"""Utilities used across other modules."""
import warnings
import numpy as np
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.activations import linear




def single_element(x):
    """If x contains a single element, return it; otherwise return x"""
    if len(x) == 1:
        x = x[0]
    return x


# This is where I wish Python had extension methods # I'll consider monkey patching it

# Model Utils

def get_nodes_by_depth(model):

    try:

        return getattr(model, '_nodes_by_depth')

    except AttributeError:

        warnings.warn("Please update keras to version 2.1.3 or greater."
                      "Support for earlier versions will be dropped in a "
                      "future release.")

        return model.nodes_by_depth


def get_model_nodes(model):

    """Return all nodes in the model"""

    return [node for v in get_nodes_by_depth(model).values() for node in v]


# Layer Utils

def get_inbound_nodes(layer):

    try:

        return getattr(layer, '_inbound_nodes')

    except AttributeError:
        
        warnings.warn("Please update keras to version 2.1.3 or greater."
                      "Support for earlier versions will be dropped in a "
                      "future release.")

        return layer.inbound_nodes


def find_nodes_in_model(model, layer):

    """Find the indices of layer's inbound nodes which are in model"""

    model_nodes = get_model_nodes(model)

    node_indices = []

    for i, node in enumerate(get_inbound_nodes(layer)):

        if node in model_nodes:

            node_indices.append(i)

    return node_indices


