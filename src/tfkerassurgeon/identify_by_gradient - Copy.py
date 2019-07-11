"""Identify which channels to delete."""

import numpy as np 

import tensorflow as tf



def get_gratient_values(model, layer, data_generator, num_of_batches = 100):
    """
    This function identifies the gradient values of the specified layer.
    
    Args:
        model: A Keras model. Note: must be compiled
        layer: The layer whose channels will be evaluated for pruning.
        data_generator: The generator used for training. Note: the Target portion is expected to be normalized between 0.0 and 1.0 (targets including negative numbers or > 1.0 will not work).
        num_of_batches: The number of batches to evaluate for before returning the results, a larger number will take longer but will provide a better result

    Returns:
        A List of the gradients values for each channel in the layer, all normalized unit normalized (1.0 to 0.0) higher values 

    Throws:
        'The target provided by the generator exceded bounds (0.0 - 1.0)' when the target output portion of the generator returned a value that was outside the 0.0 to 1.0 range
    """

    def inverse_generator(data_gen):
        """
        This function creates an iterator that gets data from data_gen and inverts the target results so that they can be used to check the gradients. 
        
        Args:
            data_gen: The generator used for training. Note: the Target portion is expected to be normalized between 0.0 and 1.0 (targets including negative numbers or > 1.0 will not work).
        
        Returns:
            A a batch of training data with the target inverted.

        Throws:
            'The target provided by the generator exceded bounds (0.0 - 1.0)' when the target output portion of the generator returned a value that was outside the 0.0 to 1.0 range

        """
        while True:
            # get the plain images
            batch = next(data_gen)

            # I should check if anything in the batch's target is negative or greater than 1.0
            if (np.amin(batch[1]) < 0.0 or np.amax(batch[1] > 1.0)):
                raise Exception('The target provided by the generator exceded bounds (0.0 - 1.0)')

            # Note: this code only works if all anweres are between 0.0 and 1.0
            inverted_answers = (batch[1] * -1.0) + 1.0

            # Yield this next batch
            yield (batch[0], np.array(inverted_answers))
    
    # Create our inverse generator 
    datagen = inverse_generator(data_generator)

    # Get the fist batch and record it's batch size for later
    batch_data = next(datagen)
    batch_size = batch_data[0].shape[0]
    output_shape_depth = len(batch_data[0].shape)

    # Start a running total, defaulte
    running_total_grad = np.zeros(([batch_size] + layer.output.shape[1:].as_list()))

    # Loop through `num_of_batches` to accumlate accurate data, more batches gives better accuracy 
    for i in range(0, num_of_batches):

        # Get a fresh batch of data
        batch_data = next(datagen)

        # I got this bit form the internet, bless their hearts : 
        # It creates a function that evaluates the gradients for a layer
        grads = model.optimizer.get_gradients(model.total_loss, layer.output)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = tf.keras.backend.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(batch_data[0], batch_data[1])
        output_grad = f(x + y + sample_weight)[0]
        # Keep a running total of the gradients over all batches
        running_total_grad = np.add(running_total_grad, output_grad).copy()


    # sum the gradients down to each filter output 
    summed_grad = running_total_grad

    # figure out if the layer is Channels Last or Channels First 
    if (layer.data_format == 'channels_last'):
        # simply sum every dimension into the last channel
        for i in range(1, output_shape_depth):
            summed_grad = np.sum(summed_grad, axis = 0)

    elif (layer.data_format == 'channels_first'):
        # TODO: test that this works right
        # sum the non batch shape into the channels portion
        for i in range(2, output_shape_depth):
            summed_grad = np.sum(summed_grad, axis = -1)
        # sum the batch size into the channels
        summed_grad = np.sum(summed_grad, axis = 0)

    else:
        # can't do anything if it's not Channels first or last since it has no channels
        raise Exception('The Layer provided did not have an appropriate data_format of either "channels_last" or "channels_first" ')

    # Get the absolute value, if ithe gradient was strongly positioves or strongly negatives then those count equaly 
    summed_grad = np.absolute(summed_grad)
    
    # normalize the Maximum Gradient to 1 by dividing everything by it    
    print(np.amin(summed_grad)) # this can be useful when debugging
    print(np.amax(summed_grad)) # this can be useful when debugging
    norm_grad = summed_grad / np.amax(summed_grad)
    
    return norm_grad


def get_prune_by_gradient(model, layer, data_generator, prune_intensity = 0.8, num_of_batches = 100):
    """
    Gets the Ids of the Fiters/Units to be pruned based on their Gradients

    Args:
        model: A Keras model. Note: must be compiled
        layer: The layer whose channels will be evaluated for pruning.
        data_generator: The generator used for training. Note: the Target portion is expected to be normalized between 0.0 and 1.0 (targets including negative numbers or > 1.0 will not work).
        prune_intensity: this number defines how intensive the pruning process is, a lower value will prune fewer outputs per time and a higher one will prune more outputs per cycle. Values < 0.0 will result in no pruning and values > 1.0 will result in maximum pruning (eventualy pruning every item).
        num_of_batches: The number of batches to evaluate for before returning the results, a larger number will take longer but will provide a better result

    Returns:
        A List of the gradients values for each channel in the layer, all normalized unit normalized (1.0 to 0.0) higher values 

    Throws:
        'The target provided by the generator exceded bounds (0.0 - 1.0)' when the target output portion of the generator returned a value that was outside the 0.0 to 1.0 range

    """

    # use the other method to get the normalized gradients
    norm_grad = get_gratient_values(model, layer, data_generator, num_of_batches)
    
    # Sort the values and then get the output indexes of the sorted items
    sorted_values = np.sort(norm_grad)
    sorted_indexes = np.argsort(norm_grad)

    # Get the gradients average and use that to determine which outputs to prune
    grad_avg = np.average(norm_grad)

    # If the average is below the prune_intensity value, then return the indexes to all outputs that are below average
    if(grad_avg < prune_intensity):
        return sorted_indexes[ np.where( sorted_values < grad_avg) ]
    # else if the average is above prune_intensity get the below average items and take the lower half but only return ones that are below prune_intensity
    else:
        # split in half the items that are below the average
        first_half_values = np.split(sorted_values[np.where(sorted_values < grad_avg)], 2)[0]
        # only return ones that are below prune intensity
        return sorted_indexes[ np.where( first_half_values < prune_intensity) ]
