"""Identify which channels to delete."""

import numpy as np 

import tensorflow as tf

class BaseSelector:
    """
    This base class defines methods to select the outputs in a layer to be pruned and returns the list of IDs for those outputs.

    """
    def __init__(self):
        # do nothing
        self.ready = True

    def get_selection(self, prune_intensity, model, layer):
        """
        Gets the Ids of the Fiters/Units to be pruned 

        Args:
            prune_intensity: is the value that determines how agressive the pruning should be. A higher Intensity value will return more output Ids to prune.
            model: the model that is being evaluated (this is used to evaluate each output that could be selected)
            layer: the layer whose outputs are being evaluated (returned IDs correispond to this layers outputs Ids)

        Returns:
            A List of the Id for each output in the layer that has been voted to be pruned

        """

        # build a list of the selected outputs 
        selected_outputs = []

        # return the selected Ids
        return selected_outputs