
# Imports

import sys
sys.path.append("../../")

import time

import numpy as np 

import tensorflow as tf

import keras

import matplotlib.pyplot as plt    

import tfkerassurgeon

from tfkerassurgeon import surgeon

#from tfkerassurgeon import identify

from tfkerassurgeon import identify_by_gradient

from tfkerassurgeon.operations import delete_channels

print(tf.__version__)


# Set some static values that can be tweaked to experiment
# Constants

batch_size = 256

epochs = 15

nb_classes = 10

keras_verbosity = 2

input_shape = (28, 28, 1)

layers = tf.keras.layers

batches_to_check = 100


# Get the MNIST Dataset

# Load the Dataset, they provided a nice helper that does all the network and downloading for you
#(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
# This is an leterantive to the MNIST numbers dataset that is a computationlally harder problem
(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

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

#datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True, vertical_flip=False)

#datagen.fit(X_train)

#datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)

Y_train_inv = (Y_train * -1.0) + 1.0

print(Y_train.shape)
print(Y_train_inv.shape)
print(Y_train[0])
print(Y_train_inv[0])

Y_train = np.array(Y_train_inv)

# load teh trained model for experimentation
model = tf.keras.models.load_model('checkpoint_pruning_test.h5')
model.summary()


# Recompile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# since we flipped the desired results, we can use a generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True, vertical_flip=False)

datagen.fit(X_train)

datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)



layer_name  = 'first_conv_7x7'
layer = model.get_layer(name=layer_name)


# Get the fist batch and record it's batch size for later
batch_data = next(datagen)

print(batch_data[1].shape)
print(batch_data[1][0])


batch_size = batch_data[0].shape[0]
output_shape_depth = len(batch_data[0].shape)

running_total_grad = np.zeros(([batch_size] + layer.output.shape[1:].as_list()))

# I got this pit form the internet, bless their hearts
grads = model.optimizer.get_gradients(model.total_loss, layer.output)
symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
f_grad = tf.keras.backend.function(symb_inputs, grads)

for i in range(0, batches_to_check):
    batch_data = next(datagen)


    x_val, y_val, sample_weight = model._standardize_user_data(batch_data[0], batch_data[1])
    output_grad = f_grad(x_val + y_val + sample_weight)[0]


            
    if(batch_size != output_grad.shape[0] or running_total_grad.shape != output_grad.shape):
        print("Error because shapes didn't match")
        print(batch_size)
        print(running_total_grad.shape)
        print(output_grad.shape)
        # skip this one, because something weird went wrong
        continue
        

    # keep a running total
    running_total_grad = np.add(running_total_grad, output_grad).copy()

    print(i)


# sum the gradients down to each output
summed_grad = np.sum(np.sum(np.sum(running_total_grad, axis = 0), axis = 0), axis = 0)

# Get the absolute value, if it was all positioves or all negatives then those count equaly 
summed_grad = np.absolute(summed_grad)

#print(summed_grad.shape)
#print(summed_grad)

# normalize the minimum gradient to 0 by subtracting it from all
min = np.amin(summed_grad)
print(min)
# I think I can just leave it, that way I will not remove outputs that are important
#norm_grad = summed_grad - min
#summed_grad = norm_grad

# normalize the Maximum Gradient to 1 by dividing everything by it
max = np.amax(summed_grad)
print(max)
norm_grad = summed_grad / max

# print the results
#print(norm_grad.shape)
#print(norm_grad)


print(np.argmin(norm_grad))
print(np.argmax(norm_grad))
print(np.average(norm_grad))


sorted_indexes = np.argsort(norm_grad)
print(sorted_indexes.shape)
print(sorted_indexes)

sorted_values = np.sort(norm_grad)
print(sorted_values.shape)
print(sorted_values)


print(np.argmin(norm_grad))
print(np.argmax(norm_grad))
print(np.average(norm_grad))


plt.violinplot(norm_grad)
plt.show()



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

datagen = inverse_generator(datagen)


norm_grad = identify_by_gradient.get_gratient_values(model, layer, datagen)


print(np.argmin(norm_grad))
print(np.argmax(norm_grad))
print(np.average(norm_grad))


sorted_indexes = np.argsort(norm_grad)
print(sorted_indexes.shape)
print(sorted_indexes)

sorted_values = np.sort(norm_grad)
print(sorted_values.shape)
print(sorted_values)


print(np.argmin(norm_grad))
print(np.argmax(norm_grad))
print(np.average(norm_grad))


plt.violinplot(norm_grad)
plt.show()


print(identify_by_gradient.get_prune_by_gradient(model, layer, datagen))



print("Done!")



