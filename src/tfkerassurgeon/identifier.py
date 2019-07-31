"""Identify which channels to delete."""
import numpy as np
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.models import Model

from tfkerassurgeon import utils


class BaseIdentifier:
    """
    This base class defines methods to identify the outputs in a layer that are candidates to be pruned, and return a list of votes, that score each Output based on prune priority

    """
    def __init__(self):
        # do nothing
        self.ready = True

    def get_votes(self, model, layer, data_generator, num_of_batches):
        """
        Gets the vote score for each output, where a higher vote means it is a better candidate for pruning.

        Args:
            model: the model that is being evaluated (this is used to evaluate each output to calculate vote)
            layer: the layer whose outputs are being evaluated (returned vote indexes correspond to this layers outputs Ids)
            data_generator: the data source to use for this evaluation (this can be called with multiple times with different datasources to build an aggregate vote)
            num_of_batches: defines how many data batches from the generator to use while calculating. (larger number is slower, but more accurate).

        Returns:
            A List of the votes for each output in the layer where the vote defines how good a prune candidate the output is

        """

        # build a list of the output's votes 
        output_votes = []

        # return the votes
        return output_votes

