
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

# Set some static values that can be tweaked to experiment
# Constants

keras_verbosity = 1

input_shape = (28, 28, 1)

nb_classes = 10

batch_size = 256

epochs = 5

num_of_full_passes = 20

cutoff_acc = 0.95

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
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True, vertical_flip=False, fill_mode = 'constant', cval = 0.0)

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

print(np.argmax(first_batch[1][0]))

plt.imshow(np.squeeze(first_batch[0][0], axis=-1), cmap='gray', interpolation='none')
plt.show()


# Clear everything from memory

#del model

tf.keras.backend.clear_session()




# Build a very small Dense net as an example

model = tf.keras.models.Sequential()

model.add(layers.AveragePooling2D((2,2),input_shape=(28, 28, 1)))

model.add(layers.Conv2D(20,
                 (3, 3),
                 activation='relu',
                 name='conv_1'))

model.add(layers.MaxPool2D())

model.add(layers.Conv2D(50, (3, 3), activation='relu', name='conv_2'))

model.add(layers.MaxPool2D())

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
          epochs=int(2*epochs),
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


# Clear everything from memory
del model

tf.keras.backend.clear_session()



model = tf.keras.models.load_model('pruned_raw_weights.h5')
model.summary()

model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


model.fit(
          X_train,
          Y_train,
          epochs=int(2*epochs),
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
print("Efficency: ", score[1]/(time.time() - start_time))











# Clear everything from memory

del model

tf.keras.backend.clear_session()




# Build a very small Dense net as an example

model = tf.keras.models.Sequential()

model.add(layers.AveragePooling2D((2,2),input_shape=(28, 28, 1)))

model.add(layers.Conv2D(20,
                 (3, 3),                 
                 activation='relu',
                 name='conv_1'))

model.add(layers.MaxPool2D())

model.add(layers.Conv2D(50, (3, 3), activation='relu', name='conv_2'))

model.add(layers.MaxPool2D())


model.add(layers.Flatten())

model.add(layers.Dense(500, activation='relu', name='dense_1'))

model.add(layers.Dense(10, activation='softmax', name='dense_2'))

model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

model.summary()


# Run the training

# One final training to make sure it fits well
model.fit_generator(datagen,
        steps_per_epoch=256,
        epochs=int(2*epochs),
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


# Clear everything from memory
del model

tf.keras.backend.clear_session()



model = tf.keras.models.load_model('pruned_raw_weights.h5')
model.summary()

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


# Then Print the Training Results
score = model.evaluate(X_test, Y_test, verbose=keras_verbosity)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Test the total time to predict the whole Validation set
start_time = time.time()
model.predict(X_test, verbose=keras_verbosity)
print("--- %s seconds ---" % (time.time() - start_time))


# Print our 'Efficency' as the Accuracy / Total Time
print("Efficency: ", score[1]/(time.time() - start_time))




