
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
file_name_prefix = "My_mnist_3_"

keras_verbosity = 0

input_shape = (28, 28, 1)

nb_classes = 10

batch_size = 256

epochs = 5

num_of_full_passes = 10

num_of_batches = 100 

cutoff_acc = 0.985

layers = tf.keras.layers


# Get the MNIST Dataset

# Load the Dataset, they provided a nice helper that does all the network and downloading for you
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
# This is an leterantive to the MNIST numbers dataset that is a computationlally harder problem
#(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

# we need to make sure that the images are normalized and in the right format
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# expand the dimensions to get the shape to (samples, height, width, channels) where greyscale has 1 channel
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# one-hot encoding, this way, each digit has a probability output
Y_train = keras.utils.np_utils.to_categorical(Y_train, nb_classes)
Y_test = keras.utils.np_utils.to_categorical(Y_test, nb_classes)

# log some basic details to be sure things loaded
print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
#print('X_train:', X_train.shape)
#print('Y_train:', Y_train.shape)
#print('X_test:', X_test.shape)
#print('Y_test:', Y_test.shape)


# Create a datagenerator 
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=False, vertical_flip=False, fill_mode = 'constant', cval = 0.0)

datagen.fit(X_train)

datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)

first_batch = next(datagen)


def grab_train_batch(batch_size, X_train, Y_train):
    # get the files iterator
    size_batch = batch_size
    last_index = len(X_train) - 1
    x_train = X_train
    y_train = Y_train

    # continue indefinitly
    while True:

        # return one batch at a time
        batch_data = [[],[]]
        for i in range(0, size_batch):
            # just grab items randomly from the training data
            random_index = random.randint(0, last_index)
            batch_data[0].append(x_train[random_index])
            batch_data[1].append(y_train[random_index])

        yield (np.array(batch_data[0]), np.array(batch_data[1]))

train_batcher = grab_train_batch(batch_size, X_train, Y_train)
first_batch = next(train_batcher)

# Lets print of the first image just to be sure everything loaded

#print(np.argmax(first_batch[1][0]))

#plt.imshow(np.squeeze(first_batch[0][0], axis=-1), cmap='gray', interpolation='none')
#plt.show()


# Build the model

input_placeholder = layers.Input(shape=input_shape)

average_pool = layers.AveragePooling2D((2,2))(input_placeholder)

altFirst_conv = layers.Conv2D(256, (3, 3), strides=(2,2), padding='same', activation='relu', name='first_conv_3x3')(average_pool)

first_half_layer = layers.Conv2D(256, (7, 1), strides=(4,1), padding='same', activation='relu', name='first_conv_7x1')( input_placeholder )
second_half_layer = layers.Conv2D(256, (1, 7), strides=(1,4), padding='same', activation='relu', name='first_conv_1x7')( first_half_layer )

first_batch = layers.Concatenate()([second_half_layer, altFirst_conv])

second_layer = layers.Conv2D(128, (1, 1), activation='relu', name='second_conv_1x1')(first_batch)

mp_2x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(second_layer)
mp_4x4 = layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same')(second_layer)
mp_6x6 = layers.MaxPooling2D(pool_size=(6, 6), strides=(2, 2), padding='same')(second_layer)
maxpool = layers.Concatenate()([mp_2x2, mp_4x4, mp_6x6])

mp1_layer = layers.Conv2D(128, (1, 1), activation='relu', name='mp1_conv_1x1')(maxpool)

next_conv = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='third_conv_3x3')(mp1_layer)

dailate_conv = layers.Conv2D(256, (3, 3), dilation_rate=2, padding='same', activation='relu', name='dial_conv_3x3')(maxpool)

second_batch = layers.Concatenate()([next_conv, dailate_conv])

squeze_conv = layers.Conv2D(128, (1, 1), activation='relu', name='fourth_conv_1x1')(second_batch)

mp2_2x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(squeze_conv)
mp2_4x4 = layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same')(squeze_conv)
mp2_6x6 = layers.MaxPooling2D(pool_size=(6, 6), strides=(2, 2), padding='same')(squeze_conv)
maxpool2 = layers.Concatenate()([mp2_2x2, mp2_4x4, mp2_6x6])

mp2_conv = layers.Conv2D(128, (1, 1), activation='relu', name='mp2_conv_1x1')(maxpool2)

flatten = layers.Flatten()(mp2_conv)

dense_layer1 = layers.Dense(128, activation='relu', name='dense_1')(flatten)

dense_layer2 = layers.Dense(256, activation='relu', name='dense_2')(dense_layer1)

dense_layer3 = layers.Dense(512, activation='relu', name='dense_3')(dense_layer2)

last_layer = layers.Dense(10, activation='softmax', name='dense_last')(dense_layer3)

model = tf.keras.models.Model(inputs=input_placeholder, outputs=last_layer) 

model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


model.summary()



# We're going to loop through the pruning process multiple times
# We're Going to prune them one layer at a time. 
model.save(file_name_prefix+'raw_weights.h5') 
model.save(file_name_prefix+'pruned_raw_weights.h5') 
model.save(file_name_prefix+'checkpoint_raw_weights.h5') 

# else load the previous run
model = tf.keras.models.load_model(file_name_prefix+'checkpoint_raw_weights.h5')
model.summary()

# Recompile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# run it through everything num_of_full_passes times for good measure
for full_pass_id in range(1, num_of_full_passes+1):

  # list off the layers we are pruning and the order that we should prune them
  prune_targets = ['second_conv_1x1','first_conv_3x3', 'first_conv_7x1','first_conv_1x7','mp1_conv_1x1','third_conv_3x3', 'dial_conv_3x3','fourth_conv_1x1','mp2_conv_1x1','dense_1','dense_2','dense_3']

  print("Starting Pass ",full_pass_id)

  # prune each layer one at a time
  for prune_target in prune_targets:

    # this will run until the exit condition is hit
    while (True):
      # Run the training
      #model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
      #    verbose=keras_verbosity, validation_data=(X_test, Y_test))
      
      start_time = time.time()
      model.fit_generator(datagen, steps_per_epoch=256, epochs=(epochs + full_pass_id), verbose=keras_verbosity,
                 max_queue_size=1000, workers=1)
      print("Fit took %s seconds" % (time.time() - start_time))

      # Then Print the Training Results
      score = model.evaluate(X_test, Y_test, verbose=keras_verbosity)
      print('Test score:', score[0])
      print('Test accuracy:', score[1])

      # check the score did not fall below the threshold, if so, undo the change
      if (score[1] < cutoff_acc):
        print("Score was below the Cutoff. ", score[1], cutoff_acc)

        # Clear everything from memory
        del model
        tf.keras.backend.clear_session()

        # load the model from the last backup
        model = tf.keras.models.load_model(file_name_prefix+'checkpoint_raw_weights.h5')
        model.save(file_name_prefix+'pruned_raw_weights.h5') 
        model.summary()

        # Recompile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        break
      # if the accuracy is good, then prune it

      # Save a backup before we prune
      model.save(file_name_prefix+'checkpoint_pruning.h5') 
      
      # Test the total time to predict the whole Validation set
      start_time = time.time()
      model.predict(X_test, verbose=keras_verbosity)
      print("--- %s seconds ---" % (time.time() - start_time))

      # Print our 'Efficency' as the Accuracy / Total Time
      print("Efficency: ", score[1]/(time.time() - start_time))






      print("Starting pruning process")

      # First we get the layer we are working on
      layer = model.get_layer( name=prune_target )

      # set the Prune intensity to slowly increase as we go further 
      prune_intensity = (float(full_pass_id) / float(num_of_full_passes))
      print("Using pruning intensity: ", prune_intensity)

      # Get the Output Indexes that are indicated as needing to be pruned
      print("Starting Inverse Gradient identification using augmented data")

      prune_layer_outputs_votes = []

      # Using my inverse Gradient Method 
      temp = identify_by_gradient.get_prune_by_gradient(model, layer, datagen, prune_intensity = prune_intensity, num_of_batches=num_of_batches)
      print(len(temp))
      prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)
      
      #print(len(prune_layer_outputs_votes))
      #print("Starting APOZ identification using augmented data")

      # Get the Channel Ids that have a high APOZ (Average Percentage of Zeros) that should identify where we can prune
      #temp = identify.high_apoz(identify.get_apoz(model, layer, datagen, num_of_batches=num_of_batches))
      #print(temp)
      #print(temp.shape)
      #prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)
      
      #print(len(prune_layer_outputs_votes))
      print("Starting Inverse Gradient identification using non-augmented data")

      # Using my inverse Gradient Method 
      temp = identify_by_gradient.get_prune_by_gradient(model, layer, train_batcher, prune_intensity = prune_intensity, num_of_batches=num_of_batches)
      print(len(temp))
      prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)
      
      #print(len(prune_layer_outputs_votes))
      print("Starting APOZ identification using non-augmented data")

      # Get the Channel Ids that have a high APOZ (Average Percentage of Zeros) that should identify where we can prune
      temp = identify.high_apoz(identify.get_apoz(model, layer, train_batcher, num_of_batches=num_of_batches))
      print(len(temp))
      prune_layer_outputs_votes = prune_layer_outputs_votes + list(temp)

      #print(len(prune_layer_outputs_votes))
      print("Finished identification for pruning process")

      # prune all items with more than 1 vote
      prune_outputs = list(set([x for x in prune_layer_outputs_votes if prune_layer_outputs_votes.count(x) > 1]))
      print(len(prune_outputs))
      print("Pruning out these layers based on votes:", prune_outputs)



      # if there are only a few outputs to prune, lets move on to the next one. 
      # as we get deeper into the pruneing loops we lower the bar
      if(len(prune_outputs) < (num_of_full_passes + 1 - full_pass_id) ):
                
        print("Outputs to prune were less than limit.", len(prune_outputs), (num_of_full_passes + 1 - full_pass_id))

        # Clear everything from memory
        del model
        tf.keras.backend.clear_session()

        # load the model from the raw weights so we can train again
        model = tf.keras.models.load_model(file_name_prefix+'checkpoint_raw_weights.h5')
        model.save(file_name_prefix+'pruned_raw_weights.h5') 
        model.summary()

        # Recompile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # break out of this loop, to do the next 
        break


      # load the raw weights model and prune from it instead
      model = tf.keras.models.load_model(file_name_prefix+'pruned_raw_weights.h5')      
      # save the previous weights as a checkpoint to go back to if exit condition is hit
      model.save(file_name_prefix+'checkpoint_raw_weights.h5') 
            
      # First we get the layer we are working on
      layer = model.get_layer(name=prune_target)

      try:

          # Run the pruning on the Model and get the Pruned (uncompiled) model as a result
          model = delete_channels(model, layer, prune_outputs)

      except Exception as ex:

        print("Could not delete layers")
       
        print(ex)

        # Clear everything from memory
        del model
        tf.keras.backend.clear_session()

        # load the model from the raw weights so we can train again
        model = tf.keras.models.load_model(file_name_prefix+'checkpoint_raw_weights.h5')
        model.save(file_name_prefix+'pruned_raw_weights.h5') 
        model.summary()

        # Recompile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # break out of this loop, to do the next 
        break
        

      # Save a the new raw weights after we prune
      model.save(file_name_prefix+'pruned_raw_weights.h5') 

      # Clear everything from memory
      del model
      tf.keras.backend.clear_session()

      # load the model from the raw weights so we can train again
      model = tf.keras.models.load_model(file_name_prefix+'pruned_raw_weights.h5')

      # Recompile the model
      model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

      print("Loop finished.")



# One final training to make sure it fits well
model.fit_generator(datagen,
        steps_per_epoch=256,
        epochs=int(2*epochs),
        verbose=keras_verbosity,
        validation_data=(X_test, Y_test)
        )


# Test the total time to predict the whole Validation set
start_time = time.time()
model.predict(X_test, verbose=keras_verbosity)
print("--- %s seconds ---" % (time.time() - start_time))


# Print our 'Efficency' as the Accuracy / Total Time
print("Efficency: ", score[1]/(time.time() - start_time))

model.save(file_name_prefix+'pruned_model.h5')
model.save_weights(file_name_prefix+'pruned_model_weights.h5')


# Clear everything from memory
del model
tf.keras.backend.clear_session()




# Build a very small Dense net as an example

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(20,
                 (3, 3),
                 input_shape=(28, 28, 1),
                 activation='relu',
                 name='conv_1'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(50, (3, 3), activation='relu', name='conv_2'))
model.add(layers.MaxPool2D())
model.add(layers.Permute((2, 1, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu', name='dense_1'))
model.add(layers.Dense(10, activation='softmax', name='dense_2'))

model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

model.summary()





# Run the training

model.fit(
          X_train,
          Y_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=keras_verbosity,
          validation_data=(X_test, Y_test)
         )


# Then Print the Training Results
score = model.evaluate(X_test, Y_test, verbose=keras_verbosity)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Test the total time to predict the whole Validation set
start_time = time.time()
model.predict(X_test, verbose=keras_verbosity)
print("--- %s seconds ---" % (time.time() - start_time))


# Print our 'Efficency' as the Accuracy / Total Time
print(score[1]/(time.time() - start_time))
