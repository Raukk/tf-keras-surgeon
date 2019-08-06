"""Identify which channels to delete."""

import numpy as np 

import tensorflow as tf

class DemocraticSelector:
    """
    This class uses the provided Identifier's votes to select the outputs to be pruned and returns the list of IDs for those outputs.

    It is designed to work with many Identifiers voting together, and Votes can be weighted to favor certain Identifiers.
    It is also designed to wor with multiple data generators (Augmented, raw, etc.) and it can be weighted to favor better data Generators (Favor Augmented data over raw data) .
    

    Arguments:
        all_voters: all the Identifiers that will be voting on which layers to prune
        all_data_generator: a List of data generator that returns random batches of training data, augmented or not (should not include test/validation data)
        num_of_batches: the number of batches to use when calculating votes, lower = faster but worse accuracy, and higher = slower but better accuracy
        voter_weights: this is a list of weights (0.0 - 1.0) to apply to each voters votes (should sum to 1.0) where index matches the voter index (None = equaliy weighted) 
        data_gen_weigths: this is a list of weights (0.0 - 1.0) to apply to the votes based on the generator used (should sum to 1.0) where index matches the voter index (None = equaliy weighted) 
    """
    def __init__(self, all_voters, all_data_generators, num_of_batches = 100, voter_weights=None, data_gen_weigths=None):
        # set all the values
        self.all_voters = all_voters
        self.all_data_generators = all_data_generators 
        self.num_of_batches = num_of_batches
        # Check that the values are valid
        if (voter_weights != None and len(voter_weights) != len(all_voters)):
            raise ValueError('`voter_weights` length must match `all_voters` if it is specified')

        self.voter_weights = voter_weights

        if (data_gen_weigths != None and len(data_gen_weigths) != len(all_data_generators)):
            raise ValueError('`data_gen_weigths` length must match `all_data_generators` if it is specified')

        self.data_gen_weigths = data_gen_weigths
        

    def __get_votes_for_generator(self, model, layer, data_generator):
        # gets the weighted votes for all voters 

        # first figure out how many outputs we are pruning 
        data_format = getattr(layer, 'data_format', 'channels_last')
        num_of_outputs_for_voting = layer.output.shape[-1]
        if data_format == 'channels_first':
            num_of_outputs_for_voting = layer.output.shape[1]

        # this holds the running total of the votes, starts as zeros
        total_votes = np.zeros(num_of_outputs_for_voting)

        for i in range(0, len(self.all_voters)):
            # run each voter one at a time, then apply the weighting
            voter = self.all_voters[i]
            votes = voter.get_votes(model, layer, data_generator, self.num_of_batches)
            # if we have a weight value for each voter, apply it now
            if(self.data_gen_weigths != None):
                votes *= self.data_gen_weigths[i]
            # otherwise apply equal weighting
            else:
                 votes /= len(self.all_voters)

            # sum all the votes together
            total_votes += votes

        # return the total (weighted) votes
        return total_votes


    def __get_votes_for_all_generators(self, model, layer):
        # gets the weighted votes for all data generators

        # first figure out how many outputs we are pruning 
        data_format = getattr(layer, 'data_format', 'channels_last')
        num_of_outputs_for_voting = layer.output.shape[-1]
        if data_format == 'channels_first':
            num_of_outputs_for_voting = layer.output.shape[1]

        # this holds the running total of the votes, starts as zeros
        total_votes = np.zeros(num_of_outputs_for_voting)

        for i in range(0, len(self.all_data_generators)):
            # run each generator one at a time, then apply the weighting
            generator = self.all_data_generators[i]
            votes = self.__get_votes_for_generator(model, layer, generator)
            # if we have a weight value for each data generator, apply it now
            if(self.data_gen_weigths != None):
                votes *= self.data_gen_weigths[i]
            # otherwise apply equal weighting
            else:
                 votes /= len(self.all_data_generators)

            # sum all the votes together
            total_votes += votes

        # return the total (weighted) votes
        return total_votes


    def get_selection(self, prune_intensity, model, layer):
        """
        Gets the Ids of the Fiters/Units to be pruned based on the votes provided by the itdentification methods

        Args:
            prune_intensity: is the value that determines how agressive the pruning should be. A higher Intensity value will return more output Ids to prune.
            model: the model that is being evaluated (this is used to evaluate each output that could be selected)
            layer: the layer whose outputs are being evaluated (returned IDs correispond to this layers outputs Ids)

        Returns:
            A List of the Id for each output in the layer that has been voted to be pruned

        """

        # build a list of the selected outputs 
        selected_outputs = []

        # get all the votes for all the data
        all_votes = self.__get_votes_for_all_generators(model, layer)

        # Sort the values and then get the output indexes of the sorted items
        sorted_indexes = np.argsort(all_votes).flip() # flip because we want highest votes first

        # this is the simplest way to use prune_intensity
        for index in sorted_indexes:
            # check if this output's votes are greater that 1.0 - the prune intensity
            # if so, add it to the list of selected. 
            if(all_votes[index] >= (1.0 - prune_intensity)):
                selected_outputs.append(index)
            #If not, then return because no more values will have higher votes
            else:
                return selected_outputs

        # return the selected Ids
        return selected_outputs