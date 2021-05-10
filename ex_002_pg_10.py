# -*- coding: utf-8 -*-
"""
Book: Advanced Deep Learning with Keras (Example, page: 10)
Author: Rowel Atienza

code modified by: jubran akram
"""

import warnings

from tensorflow import keras as krs
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

krs.backend.clear_session()

# function to load MNIST dataset that also splits the data into validation sets (if required)


def load_mnist_data(validation_set=0):
    '''This function loads mnist dataset'''
    (x_train, y_train), (x_test, y_test) = krs.datasets.mnist.load_data()
    # create validation set
    if validation_set:
        x_train, x_validation, y_train, y_validation = \
            train_test_split(x_train, y_train,
                             test_size=validation_set, shuffle=True)
    else:
        x_validation = np.array([])
        y_validation = np.array([])

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def one_hot_vector_repr(in_data):
    '''This function returns one-hot encoding of the input'''
    return krs.utils.to_categorical(in_data)


def pre_process_train_set(in_data, in_size, max_value):
    '''This function reshapes and normalizes the input'''
    in_data = np.reshape(in_data, [-1, in_size])
    return in_data.astype('float32')/max_value


def check_unique_labels(in_train):
    '''This function finds the number of unique labels'''
    unique_train = np.unique(in_train)
    return len(unique_train)

# function to automatically generate a feedforward neural network based on user's input


def feedforward_model(input_shape, number_of_hidden_layers, dropout_params,
                      number_of_neurons, activation_type,
                      kernel_regularizer, output_layer_nodes,
                      output_layer_activation):
    '''This function returns a model based on the input parameters'''

    model = krs.models.Sequential()
    for idx in range(number_of_hidden_layers):
        if idx:
            model.add(krs.layers.Dense(number_of_neurons[idx],
                                       activation=activation_type[idx],
                                       kernel_regularizer=kernel_regularizer[idx]))
        else:
            model.add(krs.layers.Dense(number_of_neurons[idx],
                                       activation=activation_type[idx],
                                       input_shape=input_shape,
                                       kernel_regularizer=kernel_regularizer[idx]))
        if dropout_params[idx]:
            model.add(krs.layers.Dropout(dropout_params[idx]))

    model.add(krs.layers.Dense(output_layer_nodes,
                               activation=output_layer_activation))

    return model


if __name__ == '__main__':

    # load mnist dataset
    x_train, y_train, x_validation, y_validation, x_test, y_test = \
        load_mnist_data(0.2)

    # training, validation and test set size
    num_train_samples, *image_dimensions = x_train.shape
    num_inputs = np.prod(image_dimensions)

    num_validation_samples = x_validation.shape[0]
    num_test_samples = x_test.shape[0]

    # resize and normalize input training, validation and test sets
    x_train = pre_process_train_set(x_train, num_inputs, 255)
    x_validation = pre_process_train_set(x_validation, num_inputs, 255)
    x_test = pre_process_train_set(x_test, num_inputs, 255)

    # unique labels in the data set
    num_labels = check_unique_labels(y_train)

    # one-hot encoding of labels
    y_train = one_hot_vector_repr(y_train)
    if y_validation.size:
        y_validation = one_hot_vector_repr(y_validation)
    y_test = one_hot_vector_repr(y_test)

    # model parameters
    input_shape = (num_inputs,)
    num_hidden_layers = 2
    dropout_params = [0.2, 0.2]
    num_neurons = [256, 256]
    activation_type = ['relu']*num_hidden_layers
    kernel_regularizer = [None, None]
    output_layer_nodes = num_labels
    output_layer_activation = 'softmax'

    # create a feedforward model
    model = feedforward_model(input_shape, num_hidden_layers, dropout_params, num_neurons,
                              activation_type, kernel_regularizer, output_layer_nodes,
                              output_layer_activation)

    print(model.summary())

    # training parameters
    loss_type = 'categorical_crossentropy'
    metrics_type = ['accuracy']
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 32

    opt = krs.optimizers.Adam(lr=learning_rate)

    # compile model
    model.compile(loss=loss_type,
                  optimizer=opt,
                  metrics=metrics_type)

    # fit model
    if x_validation.size:
        model.fit(x_train, y_train,
                  epochs=num_epochs,
                  verbose=1,
                  shuffle=True,
                  batch_size=batch_size,
                  validation_data=(x_validation, y_validation))
    else:
        model.fit(x_train, y_train,
                  epochs=num_epochs,
                  verbose=1,
                  shuffle=True,
                  batch_size=batch_size)

    # training vs. validation performance
    train_loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    train_acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()

    axs[0].plot(np.arange(num_epochs), train_loss, color='blue',
                label='Training loss', linewidth=2)
    axs[0].plot(np.arange(num_epochs), val_loss, color='red',
                label='Validation loss', linewidth=2)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training vs. Validation Loss')
    axs[0].legend()

    axs[1].plot(np.arange(num_epochs), train_acc, color='blue',
                label='Training accuracy', linewidth=2)
    axs[1].plot(np.arange(num_epochs), val_acc, color='red',
                label='Validation accuracy', linewidth=2)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training vs. Validation Accuracy')
    axs[1].legend()

    score = model.evaluate(x_test, y_test)

    print(f"Model accuracy on test data set is: {score[1]:.2f}")
    fig.suptitle(f'Test Data Accuracy:{score[1]*100:.2f}%')

    # predict labels for test data
    y_predict = model.predict(x_test)
    # check results
    ind = np.random.randint(0, num_test_samples, size=25)
    images = x_test[ind]
    labels = y_predict[ind]

    # plotting these as 5x5 grid
    fig, axs = plt.subplots(5, 5)
    axs = axs.flatten()
    for idx in range(len(axs)):
        num = np.argmax(labels[idx])
        axs[idx].imshow(images[idx].reshape(image_dimensions), cmap='gray')
        axs[idx].set_title(f'Prediction: {num}')
        axs[idx].axis('off')
    fig.suptitle('MLPs Classification Result on MNIST Data Set')
