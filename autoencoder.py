# Code executable for part A
import keras
import numpy as np
import argparse
import os

from dataset import Dataset
from keras import Model, Input, layers, optimizers
from sklearn.model_selection import train_test_split

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
        
    return layers.Conv2D(1, (convolutional_filter_size, convolutional_filter_size), activation='linear', padding='same')(new_conv)



# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d','--Dataset')
args = args_parser.parse_args()

dataset_file = args.Dataset

# Check if dataset file exists
if os.path.isfile(dataset_file):

    # User Arguments
    convolutional_layers = int(input("Number of convolutional layers: "))
    convolutional_filter_size = int(input("Convolutional filter size: "))
    convolutional_filters_per_layer = int(input("Convolutional filters per layer: "))
    epochs = int(input("Epochs: "))
    batch_size = int(input("Batch size: "))

    # Read Dataset
    dataset = Dataset(dataset_file)

    # Build the autoencoder
    x_dimension, y_dimension = dataset.getImageDimensions()
    inChannel = 1
    input_img = Input(shape=(x_dimension, y_dimension, inChannel))
    
    encoded = encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer)
    decoded = decoder(encoded, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

    # Split dataset into train and test datasets
    images = dataset.getImages()
    X_train, X_test, y_train, y_test = train_test_split(
        images,
        images,
        test_size=0.2,
        random_state=13
    )

    # Train the autoencoder
    autoencoder_train = autoencoder.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(X_test, y_test)
    )
    
    autoencoder.save_weights('autoencoder.h5')
    
else:
    print("Could not find dataset file: " + dataset_file)
