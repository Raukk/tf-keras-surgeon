
# Imports
import os

import sys

import time

import random

import numpy as np 

import tensorflow as tf

import matplotlib.pyplot as plt    

import keras




def grab_train_batch(batch_size, X_train, Y_train):
    """
    Creates a simple iterator that returns a batch of random samples from the training set
    """

    # get the data to iterate
    size_batch = batch_size
    last_index = len(X_train) - 1
    x_train = X_train
    y_train = Y_train

    # continue indefinitly, FOREVER!!!!
    while True:

        # return one batch at a time
        batch_data = [[],[]]
        for i in range(0, size_batch):
            # Just grab items randomly from the training data
            random_index = random.randint(0, last_index)
            # Append the item to the batch
            batch_data[0].append(x_train[random_index])
            batch_data[1].append(y_train[random_index])

        # return the full batch of questions and answers
        yield (np.array(batch_data[0]), np.array(batch_data[1]))


def get_mnist_data(data_format = 'channels_last', show_first_sample = False):
    """
    This method gets the MNIST data and normalizes it (X /= 255) and sets the answers to categorical
    """

    # Load the Dataset, they provided a nice helper that does all the network and downloading for you
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    # we need to make sure that the images are normalized and in the right format
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # expand the dimensions to get the shape to (samples, height, width, channels) where greyscale has 1 channel
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # We can swap the axis if it is channels first
    if data_format == 'channels_first':
        a = np.swapaxes(a, 1, -1)

    # one-hot encoding, this way, each digit has a probability output
    Y_train = keras.utils.np_utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.np_utils.to_categorical(Y_test, nb_classes)

    # We can display the first sample in the data set just to verify that everything is working
    if (show_first_sample):
        # print the first answer 
        print(np.argmax(Y_train[0]))
        # and display the first image
        plt.imshow(np.squeeze(X_train[0], axis=-1), cmap='gray', interpolation='none')
        plt.show()


    return (X_train, Y_train), (X_test, Y_test)


def get_fashion_mnist_data(data_format = 'channels_last', show_first_sample = False):
    """
    This method gets the Fashion MNIST data and normalizes it (X /= 255) and sets the answers to categorical
    """

    # Load the Dataset, keras provided a nice helper that does all the network and downloading for you
    # This is an alterantive to the MNIST numbers dataset that is a computationlally harder problem
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

    # we need to make sure that the images are normalized and in the right format
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # expand the dimensions to get the shape to (samples, height, width, channels) where greyscale has 1 channel
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # We can swap the axis if it is channels first
    if data_format == 'channels_first':
        a = np.swapaxes(a, 1, -1)

    # one-hot encoding, this way, each digit has a probability output
    Y_train = keras.utils.np_utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.np_utils.to_categorical(Y_test, nb_classes)

    # We can display the first sample in the data set just to verify that everything is working
    if (show_first_sample):
        # print the first answer 
        print(np.argmax(Y_train[0]))
        # and display the first image
        plt.imshow(np.squeeze(X_train[0], axis=-1), cmap='gray', interpolation='none')
        plt.show()
    
        return (X_train, Y_train), (X_test, Y_test)


def get_fashion_mnist_data_iterator(batch_size, data_format = 'channels_last'):
    """
    Loads the Fashion MNIST data and creates an iterator that just randomly samples the data
    """
    # get the data using the other functions
    (X_train, Y_train), (X_test, Y_test) = get_fashion_mnist_data(data_format)

    # Create an iterator that just randomly samples the data
    train_batcher = grab_train_batch(batch_size, X_train, Y_train)

    return train_batcher


def get_fashion_mnist_augmented_data_generator(batch_size, data_format = 'channels_last'):
    """
    Loads the Fashon MNIST data and creates an augmented Data Generator 
    """
    # get the data using the other functions
    (X_train, Y_train), (X_test, Y_test) = get_fashion_mnist_data(data_format)

    # Create a datagenerator using the image augmentation in keras
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True, vertical_flip=False, fill_mode = 'constant', cval = 0.0)

    datagen.fit(X_train)

    datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)

    return datagen


def get_mnist_data_iterator(batch_size, data_format = 'channels_last'):
    """
    Loads the MNIST data and creates an iterator that just randomly samples the data
    """
    # get the data using the other functions
    (X_train, Y_train), (X_test, Y_test) = get_mnist_data(data_format)

    train_batcher = grab_train_batch(batch_size, X_train, Y_train)

    return train_batcher


def get_mnist_augmented_data_generator(batch_size, data_format = 'channels_last'):
    """
    Loads the MNIST data and creates an augmented Data Generator 
    """
    # get the data using the other functions
    (X_train, Y_train), (X_test, Y_test) = get_mnist_data(data_format)

    # Create a datagenerator using the image augmentation in keras
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=False, vertical_flip=False, fill_mode = 'constant', cval = 0.0)

    datagen.fit(X_train)

    datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)

    return datagen


def get_fashion_mnist_sources(batch_size, data_format = 'channels_last'):
    """
    This simply loads the data and creates both a random iterator and an augmented Data Generator 
    """
    # Just grab the iterators and return them as an array
    iter = get_fashion_mnist_data_iterator(batch_size, data_format)
    gen = get_fashion_mnist_augmented_data_generator(batch_size, data_format)

    return [iter, gen]


def get_mnist_sources(batch_size, data_format = 'channels_last'):
    """
    This simply loads the data and creates both a random iterator and an augmented Data Generator 
    """
    # Just grab the iterators and return them as an array
    iter = get_mnist_data_iterator(batch_size, data_format)
    gen = get_mnist_augmented_data_generator(batch_size, data_format)

    return [iter, gen]


