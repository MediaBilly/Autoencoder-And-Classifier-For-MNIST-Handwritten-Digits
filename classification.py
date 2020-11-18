# Code executable for part B
import argparse
import os
import numpy as np
from imageDataset import ImageDataset
from labelDataset import LabelDataset
from encoder import encoder, encoder_layers
from keras import layers, Input, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

# User Arguments
convolutional_layers = int(input("Number of convolutional layers: "))
convolutional_filter_size = int(input("Convolutional filter size: "))
convolutional_filters_per_layer = int(input("Convolutional filters per layer: "))
fully_connected_size = int(input("Fully connected layer size: "))
epochs = int(input("Epochs: "))
batch_size = int(input("Batch size: "))
#dropout = int(input("Dropout rate: "))

# Input type construction
x_dimension, y_dimension = training_set.getImageDimensions()
inChannel = 1
input_img = Input(shape=(x_dimension, y_dimension, inChannel))

# Split dataset into train and validation datasets
X_train, X_validation, y_train, y_validation = train_test_split(
    training_set.getImagesNormalized(),
    training_labels.get_labels(),
    test_size=0.2,
    random_state=13
)

# Construct the classifier NN(input -> encoder -> Flatten -> FC -> output with 10 classes(0 - 9))
encoded = encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer)
flatten = layers.Flatten()(encoded)
fc = layers.Dense(fully_connected_size, activation='relu')(flatten)
output_layer = layers.Dense(training_labels.num_classes(), activation='softmax')(fc)

classifier = Model(input_img, output_layer)
classifier.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

# Load encoder weights
classifier.load_weights(autoencoder_weights_file, by_name=True)
classifier.summary()

# Train phase 1: Only fully connected layer weights

# Make encoder layers non trainable
for layer in classifier.layers[0 : encoder_layers(convolutional_layers)]:
    layer.trainable = False

classifier_trained_phase1 = classifier.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_validation, y_validation)
)

# Train phase 2: All layer weights

# Make encoder layers trainable
for layer in classifier.layers[0:encoder_layers(convolutional_layers)]:
    layer.trainable = True

classifier_trained_phase2 = classifier.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_validation, y_validation)
)

# print(classifier.evaluate(X_validation, y_validation, batch_size=batch_size))


y_pred1 = classifier.predict(testset.getImagesNormalized(),batch_size=batch_size)
y_pred = np.argmax(y_pred1,axis=1)
print(classification_report(y_true=test_labels.get_labels(),y_pred=y_pred))