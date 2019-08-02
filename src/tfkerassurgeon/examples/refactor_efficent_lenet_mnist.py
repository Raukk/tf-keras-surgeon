
# Imports
import os

import logging

import sys

import time

import random

import numpy as np 

import tensorflow as tf

import matplotlib.pyplot as plt    

import keras

from example_data_loader import get_mnist_sources

import tfkerassurgeon

from tfkerassurgeon import surgeon

from tfkerassurgeon import identify_by_gradient

from tfkerassurgeon import identify

from tfkerassurgeon.operations import delete_channels

# set basic values
print(tf.__version__)

sys.path.append("../../")

logging.getLogger('tensorflow').disabled = True

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)


#sess = tf.Session(config=config)  #With the two options defined above

#with tf.Session() as sess:


# Set some static values that can be tweaked to experiment
# Constants

file_name_prefix = "Refactor_1_"

keras_verbosity = 0

input_shape = (28, 28, 1)

nb_classes = 10

batch_size = 256

epochs = 5

num_of_full_passes = 10

cutoff_acc = 0.99

num_of_batches = 10 # This is probably way to low to get a good value

layers = tf.keras.layers



# Get the Data we are training on.
data_sources = get_mnist_sources(batch_size) 

datagen = data_sources[1]


# Build the model

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(512, (3, 3), input_shape=(28, 28, 1), activation='relu', name='conv_1'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(256, (3, 3), activation='relu', name='conv_2'))
model.add(layers.MaxPool2D())
#model.add(layers.Permute((2, 1, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', name='dense_1'))
model.add(layers.Dense(10, activation='softmax', name='dense_2'))



def compile(model):

    new_adam = tf.keras.optimizers.Adam(clipnorm= 1.0)

    model.compile(optimizer=new_adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


# We're going to loop through the pruning process multiple times
# We're Going to prune them one layer at a time. 
model.save(file_name_prefix+'raw_weights.h5') 
model.save(file_name_prefix+'pruned_raw_weights.h5') 
model.save(file_name_prefix+'checkpoint_raw_weights.h5') 

# else load the previous run
#model = tf.keras.models.load_model(file_name_prefix+'checkpoint_raw_weights.h5')
model.summary()

# Recompile the model
compile(model)




def train_and_test(model, full_pass_id):

    # we time it as well to keep track of how th training is going
    start_time = time.time()
    #model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
    #    verbose=keras_verbosity, validation_data=(X_test, Y_test))
    model.fit_generator(datagen, steps_per_epoch=256, epochs=(epochs + full_pass_id), verbose=keras_verbosity,
            max_queue_size=1000, workers=1)
    print("Fit took %s seconds" % (time.time() - start_time))

    # Then Print the Training Results
    score = model.evaluate(X_test, Y_test, verbose=keras_verbosity)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    return score


def model_reload():
    # Clear everything from memory
    tf.keras.backend.clear_session()

    # load the model from the raw weights so we can train again
    model = tf.keras.models.load_model(file_name_prefix+'pruned_raw_weights.h5')
    model.summary()

    # Recompile the model
    compile(model)
    return model


def model_reload_prev():
    # Clear everything from memory
    tf.keras.backend.clear_session()

    # load the model from the last backup
    model = tf.keras.models.load_model(file_name_prefix+'checkpoint_raw_weights.h5')
    model.save(file_name_prefix+'pruned_raw_weights.h5') 
    model.summary()

    # Recompile the model
    compile(model)
    return model


def calc_prune_outputs(model, prune_target):
    
    # First we get the layer we are working on
    layer = model.get_layer(name = prune_target)
     
    # set the Prune intensity to slowly increase as we go further 
    prune_intensity = (float(full_pass_id) / float(num_of_full_passes))
    print("Using pruning intensity: ", prune_intensity)

    # Get the Output Indexes that are indicated as needing to be pruned
    print("Starting Inverse Gradient identification using augmented data")

    prune_layer_outputs_votes = []

    # Using my inverse Gradient Method 
    temp = identify_by_gradient.get_prune_by_gradient(model, layer, datagen, prune_intensity = prune_intensity, num_of_batches = num_of_batches)
    print(len(temp))
    prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)
      
    #print(len(prune_layer_outputs_votes))
    #print("Starting APOZ identification using augmented data")

    # Get the Channel Ids that have a high APOZ (Average Percentage of Zeros) that should identify where we can prune
    #temp = identify.high_apoz(identify.get_apoz(model, layer, datagen, num_of_batches = num_of_batches))
    #print(len(temp))
    #prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)
      
    #print(len(prune_layer_outputs_votes))
    print("Starting Inverse Gradient identification using non-augmented data")

    # Using my inverse Gradient Method 
    temp = identify_by_gradient.get_prune_by_gradient(model, layer, train_batcher, prune_intensity = prune_intensity, num_of_batches = num_of_batches)
    print(len(temp))
    prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)
      
    #print(len(prune_layer_outputs_votes))
    print("Starting APOZ identification using non-augmented data")

    # Get the Channel Ids that have a high APOZ (Average Percentage of Zeros) that should identify where we can prune
    temp = identify.high_apoz(identify.get_apoz(model, layer, train_batcher, num_of_batches = num_of_batches))
    print(len(temp))
    prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)

    #print(len(prune_layer_outputs_votes))
    print("Finished identification for pruning process")

    # prune all items with more than 1 vote
    prune_outputs = list(set([x for x in prune_layer_outputs_votes if prune_layer_outputs_votes.count(x) > 1]))
    print(len(prune_outputs))
    print("Pruning out these layers based on votes:", prune_outputs)

    return prune_outputs





def run_passes(model, num_of_passes):

    last_prune = "";

    # run it through everything num_of_full_passes times for good measure
    for full_pass_id in range(1, num_of_passes + 1):

        print("Starting Pass ", full_pass_id)

        # list off the layers we are pruning and the order that we should prune them
        prune_targets = ['dense_1', 'conv_2', 'conv_1']

        # This will run until the exit condition is hit for each later
        while (len(prune_targets) > 0):

            # prune each layer one at a time, round robin
            for prune_target in prune_targets:

                # Run the training first 
                score = train_and_test(model, full_pass_id)

                # check the score did not fall below the threshold, if so, undo the prune
                if (score[1] < cutoff_acc):
                    print("Score was below the Cutoff. ", score[1], cutoff_acc)

                    # The Pruning hit a failure case, so do the failure logic
                    prune_targets.remove(last_prune)
                    del model
                    model = model_reload_prev()

                    break
                
                # if the accuracy is good, then prune it
                print("Starting pruning process")

                # Save a backup before we prune
                model.save(file_name_prefix+'checkpoint_pruning.h5') 
      
                # identify what to prune
                to_prune = calc_prune_outputs(model, prune_target)

                prune_outputs = to_prune

                # if there are only a few outputs to prune, lets move on to the next one. 
                # as we get deeper into the pruneing loops we lower the bar
                if(len(prune_outputs) < (num_of_full_passes + 1 - full_pass_id) ):
                
                    print("Outputs to prune were less than limit.", len(prune_outputs), (num_of_full_passes + 1 - full_pass_id))
                    
                    # The Pruning hit a failure case, so do the failure logic
                    prune_targets.remove(prune_target)
                    del model
                    model = model_reload()

                    # break out of this loop, to do the next 
                    break


                # load the raw weights model and prune from it instead
                model = tf.keras.models.load_model(file_name_prefix+'pruned_raw_weights.h5')      
                # save the previous weights as a checkpoint to go back to if exit condition is hit
                model.save(file_name_prefix+'checkpoint_raw_weights.h5') 
            
                try:

                    # First we get the layer we are working on
                    layer = model.get_layer(name=prune_target)

                    # Run the pruning on the Model and get the Pruned (uncompiled) model as a result
                    model = delete_channels(model, layer, prune_outputs)

                except Exception as ex:
                    print("Error trying to delete layers")
                    print(ex)

                    # Clear everything from memory
                    prune_targets.remove(prune_target)
                    del model
                    model = model_reload_prev()

                    # break out of this loop, to do the next 
                    break        

                # Save a the new raw weights after we prune
                model.save(file_name_prefix+'pruned_raw_weights.h5') 

                # Clear everything from memory
                del model
                model = model_reload()
                last_prune = prune_target
                print("Loop finished.")

                # all done with the loop now



# Run the actual process now
run_passes(model, num_of_full_passes) 
