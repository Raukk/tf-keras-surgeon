
# Imports
import os

import logging

import sys

import time

import numpy as np 

import tensorflow as tf

import matplotlib.pyplot as plt    

import keras

import tfkerassurgeon

from tfkerassurgeon import surgeon

from tfkerassurgeon import identify_by_gradient

from tfkerassurgeon.operations import delete_channels

# set basic values
print(tf.__version__)

sys.path.append("../../")

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Set some static values that can be tweaked to experiment
# Constants

batch_size = 256

epochs = 15

nb_classes = 10

keras_verbosity = 2

input_shape = (28, 28, 1)

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
print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)



datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=False, vertical_flip=False)

datagen.fit(X_train)

datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)

first_batch = next(datagen)




# Lets print of the first image just to be sure everything loaded

print(np.argmax(first_batch[1][0]))

plt.imshow(np.squeeze(first_batch[0][0], axis=-1), cmap='gray', interpolation='none')



# Build a standard baseline model (LeNet)

input_placeholder = layers.Input(shape=input_shape)

average_pool = layers.AveragePooling2D((2,2))(input_placeholder)

first_layer = layers.Conv2D(512, (7, 7), strides=(3,3), padding='same', activation='relu', name='first_conv_7x7')(average_pool)

second_layer = layers.Conv2D(256, (1, 1), activation='relu', name='second_conv_1x1')(first_layer)

maxpool = layers.MaxPool2D((2,2))(second_layer)

next_conv = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='third_conv_3x3')(maxpool)

squeze_conv = layers.Conv2D(128, (1, 1), activation='relu', name='fourth_conv_1x1')(next_conv)

maxpool2 = layers.MaxPool2D((2,2))(squeze_conv)

flatten = layers.Flatten()(maxpool2)

dense_layer1 = layers.Dense(512, activation='relu', name='dense_1')(flatten)

dense_layer2 = layers.Dense(512, activation='relu', name='dense_2')(dense_layer1)

dense_layer3 = layers.Dense(512, activation='relu', name='dense_3')(dense_layer2)

last_layer = layers.Dense(10, activation='softmax', name='dense_last')(dense_layer3)

model = tf.keras.models.Model(inputs=input_placeholder, outputs=last_layer) 

model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


model.summary()



# We're going to loop through the pruning process multiple times
# We're Going to prune them one layer at a time. 
#model.save('raw_weights.h5') 
#model.save('pruned_raw_weights.h5') 
# else load the previous run
model = tf.keras.models.load_model('checkpoint_raw_weights.h5')
model.summary()

# Recompile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# run it through everything 10 times for good measure
for throwaway in range(0,10):

  # list off the layers we are pruning and the order that we should prune them
  prune_targets = ['first_conv_7x7','second_conv_1x1','third_conv_3x3','fourth_conv_1x1','dense_1','dense_2','dense_3']

  # prune each layer one at a time
  for prune_target in prune_targets:

    # this will run until the exit condition is hit
    while (True):
      # Run the training
      #model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
      #    verbose=keras_verbosity, validation_data=(X_test, Y_test))

      model.fit_generator(datagen, steps_per_epoch=256, epochs=epochs, verbose=keras_verbosity,
                 max_queue_size=10000, workers=1)

      # Then Print the Training Results
      score = model.evaluate(X_test, Y_test, verbose=keras_verbosity)
      print('Test score:', score[0])
      print('Test accuracy:', score[1])

      # check the score did not fall below the threshold, if so, undo the change
      if (score[1] < 0.95 ):
        # Clear everything from memory
        del model
        tf.keras.backend.clear_session()

        # load the model from the last backup
        model = tf.keras.models.load_model('checkpoint_raw_weights.h5')
        model.save('pruned_raw_weights.h5') 
        model.summary()

        # Recompile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        break
      # if the accuracy is good, then prune it

      # Save a backup before we prune
      model.save('checkpoint_pruning.h5') 
      
      # Test the total time to predict the whole Validation set
      start_time = time.time()
      model.predict(X_test, verbose=keras_verbosity)
      print("--- %s seconds ---" % (time.time() - start_time))

      # Print our 'Efficency' as the Accuracy / Total Time
      print(score[1]/(time.time() - start_time))

      # First we get the layer we are working on
      layer = model.get_layer(name=prune_target)

      # Get the Output Indexes that are indicated as needing to be pruned
      prune_outputs = identify_by_gradient.get_prune_by_gradient(model, layer, datagen)

      if(len(prune_outputs) <= 5):
                
        # Clear everything from memory
        del model
        tf.keras.backend.clear_session()

        # load the model from the raw weights so we can train again
        model = tf.keras.models.load_model('checkpoint_raw_weights.h5')
        model.save('pruned_raw_weights.h5') 
        model.summary()

        # Recompile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # break out of this loop, to do the next 
        break


      # load the raw weights model and prune from it instead
      model = tf.keras.models.load_model('pruned_raw_weights.h5')      
      # save the previous weights as a checkpoint to go back to if exit condition is hit
      model.save('checkpoint_raw_weights.h5') 
            
      # First we get the layer we are working on
      layer = model.get_layer(name=prune_target)

      # Run the pruning on the Model and get the Pruned (uncompiled) model as a result
      model = delete_channels(model, layer, prune_outputs)

      # Save a the new raw weights after we prune
      model.save('pruned_raw_weights.h5') 

      # Clear everything from memory
      del model
      tf.keras.backend.clear_session()

      # load the model from the raw weights so we can train again
      model = tf.keras.models.load_model('pruned_raw_weights.h5')

      # Recompile the model
      model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


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
print(score[1]/(time.time() - start_time))

model.save('pruned_model.h5')
model.save_weights('pruned_model_weights.h5')


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
