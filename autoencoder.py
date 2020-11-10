# Code executable for part A
#import keras
#import numpy as np
import argparse
import os

from dataset import Dataset
#from keras import layers, optimizers, losses, metrics

# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d','--Dataset')
args = args_parser.parse_args()

dataset_file = args.Dataset

# Check if dataset file exists
if os.path.isfile(dataset_file):

    # User Arguments
    convolutional_layers = input("Number of convolutional layers: ")
    convolutional_filter_size = input("Convolutional filter size: ")
    convolutional_filters_per_layer = input("Convolutional filters per layer: ")
    epochs = input("Epochs: ")
    batch_size = input("Batch size: ")

    # Read Dataset
    dataset = Dataset(dataset_file)

    # Build the NN
    #model = keras.Sequential()

else:
    print("Could not find dataset file:" + dataset_file)
