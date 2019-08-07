"""Identify which channels to delete."""

import numpy as np 

import tensorflow as tf


class InverseGradientIdentifier:
    """
    This class uses the Gradient of the inverse answers to identify which layer outputs are good candidates for pruning, 
    and return a list of votes, that score each Output based on prune priority

    Arguments:
        modifier (Optional): applied to the output, it is a power (allowing better distinction between items that are 95% zeros and 75% zeros)
            A larger number will have the effect that the vote magnitude will drop off rapidly as the % of zeros decreases (Defaults to 1.0)
    """
    def __init__(self, prune_intensity):
        #  Set the static values
        self.prune_intensity = prune_intensity

    def get_votes(self, model, layer, data_generator, num_of_batches = 10):
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

        return self.get_gratient_values(model, layer, data_generator, num_of_batches)

    def get_gratient_values(self, model, layer, data_generator, num_of_batches = 10):
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
        output_shape_depth = len(layer.output.shape)

        # Start a running total, defaulte
        running_total_grad = np.zeros(([batch_size] + layer.output.shape[1:].as_list()))

        # I got this bit from the internet, bless their hearts : 
        # It creates a function that evaluates the gradients for a layer
        grads = model.optimizer.get_gradients(model.total_loss, layer.output)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f_grad = tf.keras.backend.function(symb_inputs, grads)

        # Loop through `num_of_batches` to accumlate accurate data, more batches gives better accuracy 
        for i in range(0, num_of_batches):

            # Get a fresh batch of data
            batch_data = next(datagen)

            # Run for this batch
            x_val, y_val, sample_weight = model._standardize_user_data(batch_data[0], batch_data[1])
            output_grad = f_grad(x_val + y_val + sample_weight)[0]
        
            if(batch_size != output_grad.shape[0] or running_total_grad.shape != output_grad.shape):
                #print("Error because shapes didn't match")
                #print(batch_size)
                #print(running_total_grad.shape)
                #print(output_grad.shape)
                # skip this one, because something weird went wrong
                continue
        
            # Keep a running total of the gradients over all batches
            running_total_grad = np.add(running_total_grad, output_grad).copy()


        # sum the gradients down to each filter output 
        summed_grad = running_total_grad.copy()

        if(output_shape_depth > 2):
            if (hasattr(layer, 'data_format')): 
                # figure out if the layer is Channels Last or Channels First 
                if (layer.data_format == 'channels_last'):
                    # simply sum every dimension into the last channel
                    for i in range(2, output_shape_depth):
                        summed_grad = np.sum(summed_grad, axis = 0).copy()

                elif (layer.data_format == 'channels_first'):
                    # TODO: test that this works right
                    # sum the non batch shape into the channels portion
                    for i in range(2, output_shape_depth):
                        summed_grad = np.sum(summed_grad, axis = -1).copy()
                    # sum the batch size into the channels

                else:
                    # can't do anything if it's not Channels first or last since it has no channels
                    raise Exception('The Layer provided did not have an appropriate data_format of either "channels_last" or "channels_first" ')
            else:
                # simply sum every dimension into the last channel
                for i in range(2, output_shape_depth):
                    summed_grad = np.sum(summed_grad, axis = 0).copy()

        # sum the batch size into the outputs
        summed_grad = np.sum(summed_grad, axis = 0).copy()

        # Get the absolute value, if ithe gradient was strongly positioves or strongly negatives then those count equaly 
        summed_grad = np.absolute(summed_grad).copy()
    
        # normalize the Maximum Gradient to 1.0 by dividing everything by it    
        #print(np.amin(summed_grad)) # this can be useful when debugging#
        #print(np.amax(summed_grad)) # this can be useful when debugging#
        norm_grad = summed_grad / np.amax(summed_grad)
    
        return norm_grad


    def get_prune_by_gradient(self, model, layer, data_generator, prune_intensity = 0.8, num_of_batches = 100):
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
        norm_grad = self.get_gratient_values(model, layer, data_generator, num_of_batches)
    
        # Sort the values and then get the output indexes of the sorted items
        sorted_values = np.sort(norm_grad)
        sorted_indexes = np.argsort(norm_grad)

        # Get the gradients average and use that to determine which outputs to prune
        grad_avg = np.average(norm_grad)

        print('gradient average :', grad_avg)

        # If the average is below the prune_intensity value, then return the indexes to all outputs that are below average
        if(grad_avg < prune_intensity):
            # only return ones that are below the average
            return sorted_indexes[ np.where( sorted_values < grad_avg) ]
        # else if the average is above prune_intensity get the below average items and take the lower half but only return ones that are below prune_intensity
        else:
            # split in half the items that are below the average
            to_split = sorted_values[np.where(sorted_values < grad_avg)]
            # check that the size of stuff to split is not divisable by 2, then remove the last item
            if(len(to_split) % 2 == 1):
                to_split = to_split[:-1]

            # If there is only one result, return it if it's below prune_intensity 
            if(len(to_split) < 2):
                return sorted_indexes[ np.where( to_split < prune_intensity) ]

            # split and get just the first half of the values
            first_half_values = np.split(to_split, 2)[0]

            # only return ones that are below prune intensity
            return sorted_indexes[ np.where( first_half_values < prune_intensity) ]
