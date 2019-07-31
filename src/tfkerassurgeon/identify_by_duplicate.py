"""Identify which channels to delete."""

import time

import numpy as np 

import numba
from numba import jit

import tensorflow as tf

from tfkerassurgeon import utils


#TODO: FIGURE OUT WHERE THIS IS SLOW AND IF I CAN FIX IT


#TODO .get_votes(model, layer, data_generator, num_of_batches)



def get_batch_dup_scores (activations, non_channel_axis, check_for_inverse_corr, channels_count):


    # next, normalize all the activations per output in that batch to between -1.0 and 1.0
    # to do this we first get the max of each filters activation
    # We get the absolute value of the activations (so we can normalize based on the most positive, or most negative)
    abs_activation = np.abs(activations)
    # next we get the max but it keeps the dimensions (this lets us deivide by it) (a 3,3,16 would become 1,1,16)
    max_of_filters = np.amax(abs_activation, axis = non_channel_axis, keepdims=True)
        
    # finally we divide each filter by it's max to get the normalized activations of this batch
    # TODO: figure out what we do with divide by zeros, or to slve it
    normalized = activations / max_of_filters
        
    # we will then create a grid of the correlations between each output and every other output
    correlation_grid = []

    # We need to split out each outputs values so we can compare with every other value
    split = np.split(normalized, normalized.shape[-1], axis=-1)

    # The split gives us a numpy array for each output whose last dimension is of size 1
    for one_output_vals in split:
        # Then compare the normalized activations of all output values to every other output value for each instance
        # The magnitude of their differences across all the samples is the inverse of their corelation (lower values means more correlated)
        # So, a pair of outputs that always have a large difference in activation value are very non-corelated 

        # we simply use subtract to get the differences, and abs value to get the magnitude
        # we sum together for all axis except the last one (channels)
        pos_diff_sum = np.sum(np.abs(normalized - one_output_vals), axis = non_channel_axis)
        # sutract 1.25x time # abs 0.98x time # sum 0.56x time

        # if we aren't doing inverse checks, then we're done, this is the total magnitude of differences
        min_diff = pos_diff_sum
                        
        # If we want to also check for inverse (negative) correlations 
        if (check_for_inverse_corr):
        
            # do the exact same but with the inverse of the split outputs values
            neg_diff_sum = np.sum(np.abs(normalized - -one_output_vals), axis = non_channel_axis)
                
            # then we need to get the value for each correlation that is lowest (most correlated) between the inverse and the normal
            min_diff = np.minimum(pos_diff_sum, neg_diff_sum)

        # add it to our grid
        correlation_grid.append(min_diff)


    # make the finished grid a numpy array for easy use
    correlation_grid = np.array(correlation_grid)
    # this is for debugging
    #print(correlation_grid)
    # make a working copy for us to modify as we work through it.
    working_correlation_grid = correlation_grid.copy()
    # the result should be an x^2  mirrored (half)grid of correlated pairs, which we can order by their corelation values
        
    # This section is going to order each outputs iteratively, for every output except the last pair (since whe don't know which is better)
    # this will hold the order of outputs sorted by correlation
    ordered_output = []
    for iter in range(0, correlation_grid.shape[0] - 1):

        # ordering smallest value pairs first (most correlated) we can work through them one pair at at time
        arg_sorted = np.argsort(working_correlation_grid, axis=None)
        sorted_indexes = np.unravel_index(arg_sorted, working_correlation_grid.shape) 
        stacked = np.stack(sorted_indexes, axis=-1) # returns an array of Nx2 where N is the number of outputs and the 2 values are the indexes
        for index_pair in stacked:
            # we skip any that are the identity value in the grid (1,1 or 2,2, etc)
            if(index_pair[0] != index_pair[1]):
                # we get the value of correlation of the 2
                value = working_correlation_grid[index_pair[0] ] [ index_pair[1]]

                # we exclude any where the value is not finite (should be nan, because it's already been processed)
                if(np.isfinite( value) ):

                    # special case for the last pair, add both, and we're done
                    if(iter >= correlation_grid.shape[0]-2):
                        ordered_output.append(index_pair[0])
                        ordered_output.append(index_pair[1])
                        break


                    # we need to decide which of the 2 is most correlated with the remaining values 
                    first_sum = np.nansum(working_correlation_grid[index_pair[0]])
                    second_sum = np.nansum(working_correlation_grid[index_pair[1]])
                        
                    # the sum that is lower will be more correlated to the remaining outputs
                    if(first_sum < second_sum):
                        # we will remove the most correlated of the pair and replace all of it's pairs with nan in the working grid
                        working_correlation_grid[index_pair[0]] = np.full((working_correlation_grid.shape[0],), np.nan)
                        working_correlation_grid[:, index_pair[0]] = np.nan
                        # add it to the output and break out of the loops so we hit the next iteration
                        ordered_output.append(index_pair[0])
                        break
                    else:
                        # we will remove the most correlated of the pair and replace all of it's pairs with nan in the working grid
                        working_correlation_grid[index_pair[1]] = np.full((working_correlation_grid.shape[0],), np.nan)
                        working_correlation_grid[:, index_pair[1]] = np.nan
                        # add it to the output and break out of the loops so we hit the next iteration
                        ordered_output.append(index_pair[1])
                        break
        
    # wow that was a lot of convoluted code
    # maybe I should refactor it...

    # now we just have to score each one and sort the returns
    scores = np.zeros(channels_count)
        
    working_ordered_output = ordered_output.copy()
        
    # grab the first 2 and process their scores on all the others 
    item1 = working_ordered_output.pop()
    item2 = working_ordered_output.pop()
        
    # I need to normalize and invert the correlation_grid 
    correlation_grid /= np.nanmax(correlation_grid)
    correlation_grid = np.ones(correlation_grid.shape) - correlation_grid
        
    # get the score for this first pair and assign it to both
    pair_score = correlation_grid[item1][item2]
    scores[item1] = pair_score
    scores[item2] = pair_score

    # process the first 2, using their own score on eachother, then processing them against the rest normally
    temp = [[item1, item2], working_ordered_output]
    for list in temp:
        # we need to remove each item as we process it so that it does not gain any more score 
        while(list):
            # process them in order, each item
            item1 = list.pop()
            for item2 in working_ordered_output:
                # Get the normalized correlation score for this pair, and then append it to the running score
                pair_score = correlation_grid[item1][item2]
                scores[item2] += pair_score            

    # normalize the scores
    # Maybe not normalize here? 
    #scores /= np.max(scores)

    return scores



def get_duplicate_scores(model, layer, data_generator, batch_size, num_of_batches = 100, check_for_inverse_corr = False):
    """
    This function identifies the duplicate scores of the specified layer.
    
    Args:
        model: A Keras model. Note: must be compiled
        layer: The layer whose channels will be evaluated for pruning.
        data_generator: The generator used for training. 
        num_of_batches: The number of batches to evaluate for before returning the results, a larger number will take longer but will provide a better result

    Returns:
        A List of the duplicate scores for each channel in the layer, all unit normalized (1.0 to 0.0)

    """
    # get the Layers Data Formatt
    data_format = getattr(layer, 'data_format', 'channels_last')

    # Get the list of axis excluding the last axis (aka the channels) as a tuple, this is used in a few places
    non_channel_axis = tuple(range(0, len(layer.output_shape)-1))

    channels_count = 0
    if(data_format == 'channels_first'):
        channels_count = layer.output_shape[1]
    elif(data_format == 'channels_last'):
        channels_count = layer.output_shape[-1]

    batch_scores = np.zeros(channels_count)

    # create a keras function to get the Activation  
    get_activations = tf.keras.backend.function([model.input, tf.keras.backend.learning_phase()], layer.output)

    # Loop through `num_of_batches` to accumlate accurate data, more batches gives better accuracy 
    for i in range(0, num_of_batches):

        # Get a fresh batch of data
        batch_data = next(data_generator)

        # get the activations for a batch
        activations = get_activations([batch_data[0], 0])

        # Ensure that the channels axis is last
        if data_format == 'channels_first':
            activations = np.swapaxes(activations, 1, -1)


        
        batch_start_time = time.time()

        # run the other method
        scores = get_batch_dup_scores (activations, non_channel_axis, check_for_inverse_corr, channels_count)

        print("Per Batch --- %s seconds ---" % (time.time() - batch_start_time))


        # keep a running total across all the batches
        batch_scores += scores


    # normalize for all the batches
    batch_scores /= np.max(batch_scores)

    # For debugging
    print(np.argmax(batch_scores))

    return batch_scores    
