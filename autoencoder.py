# Code executable for part A
import keras
import numpy as np
import argparse
import os

from experiment import Experiment
from dataset import Dataset
from utility import *
from keras import Model, Input, layers, optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer):
    conv = input_img

    for layer in range(convolutional_layers):
        conv = layers.Conv2D(convolutional_filters_per_layer * (2 ** layer), (convolutional_filter_size, convolutional_filter_size), activation='relu', padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        if layer <= 1:
            conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        
    return conv


def decoder(conv, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer):
    new_conv = conv
    
    for layer in range(convolutional_layers - 1, -1, -1):
        new_conv = layers.Conv2D(convolutional_filters_per_layer * (2 ** layer), (convolutional_filter_size, convolutional_filter_size), activation='relu', padding='same')(new_conv)
        new_conv = layers.BatchNormalization()(new_conv)
        if layer <= 1:
            new_conv = layers.UpSampling2D((2, 2))(new_conv)
        
    return layers.Conv2D(1, (convolutional_filter_size, convolutional_filter_size), activation='sigmoid', padding='same')(new_conv)



repeat = True
# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d', '--Dataset')
args = args_parser.parse_args()

dataset_file = args.Dataset

# Check if dataset file exists
if os.path.isfile(dataset_file):
    # Read Dataset
    dataset = Dataset(dataset_file)

    # Build the autoencoder
    x_dimension, y_dimension = dataset.getImageDimensions()
    inChannel = 1
    input_img = Input(shape=(x_dimension, y_dimension, inChannel))

    # Load images from dataset and normalize their pixels in range [0,1]
    images_normed = dataset.getImagesNormalized()

    # Split dataset into train and validation datasets
    X_train, X_validation, y_train, y_validation = train_test_split(
        images_normed,
        images_normed,
        test_size=0.2,
        random_state=13
    )
    
    experiments = list()
    
    while repeat:
        # User Arguments
        convolutional_layers = int(input("Number of convolutional layers: "))
        convolutional_filter_size = int(input("Convolutional filter size: "))
        convolutional_filters_per_layer = int(input("Convolutional filters per layer: "))
        epochs = int(input("Epochs: "))
        batch_size = int(input("Batch size: "))

        encoded = encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer)
        decoded = decoder(encoded, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer)
        
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

        # Train the autoencoder
        autoencoder_train = autoencoder.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_validation, y_validation)
        )
        
        # Save experiment results for later use
        experiments.append(
            Experiment(
                convolutional_layers, 
                convolutional_filter_size, 
                convolutional_filters_per_layer, 
                epochs, batch_size, 
                autoencoder_train.history
            )
        )
        
        # Prompt to plot experiments
        if get_user_answer_boolean("Show loss graph(Y/N)? "):
            for index, experiment in enumerate(experiments):
                fig = plt.subplot(len(experiments), 1, index + 1)
                experiment.plot()
            
        # Save trained model weights
        if get_user_answer_boolean("Save trained model (Y/N)? "):
            save_file_path = input("Input save file path: ")
            autoencoder.save_weights(save_file_path)
        
        repeat = get_user_answer_boolean("Repeat Experiment (Y/N)? ")
        
else:
    print("Could not find dataset file: " + dataset_file)
