# Code executable for part B
import argparse
import os
from imageDataset import ImageDataset
from labelDataset import LabelDataset

repeat = True
# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d',    '--training_set')
args_parser.add_argument('-dl',   '--training_labels')
args_parser.add_argument('-t',    '--testset')
args_parser.add_argument('-tl',   '--test_labels')
args_parser.add_argument('-model','--autoencoder_h5')
args = args_parser.parse_args()

training_set_file = args.training_set
training_labels_file = args.training_labels
testset_file = args.testset
test_labels_file = args.test_labels
autoencoder_weights_file = args.autoencoder_h5

# Read training set
training_set = ImageDataset(training_set_file)

# Read training labels
training_labels = LabelDataset(training_labels_file)

# Read testset
testset = ImageDataset(testset_file)

# Read test labels
test_labels = LabelDataset(test_labels_file)
