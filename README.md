# Autoencoder-And-Classifier-For-MNIST-Handwritten-Digits

> The purpose of this project is to build a Convolutional Autoencoder Neural Network for the MNIST hand-written digit dataset, save it's weights and use the decoder part with an 
extra Fully-Connected layer to build, train and evaluate a Classifier Neural Network for those hand-written digits.

## A. Convolutional Autoencoder:
Usage: `python autoencoder.py –d <dataset>` \
The program will prompt the user to enter the hyperparameters for the Neural Network. After the training is done, the program will prompt the user if he wants to 
repeat the experiment with other hyperparameters, show loss graphs of all the experiments done, or save the model from the last experiment(in this case, the user
will be prompted to enter the file path for the file to be saved).

## B. Classifier:
Usage: `python  classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>`
This model is trained in 2 phases. In the first phase, only the fully connected layer is trained, and in the second, the whole model is trained. This strategy is
used for faster training of the model.
The program will prompt the user to enter the hyperparameters for the Neural Network. After the training is done, the program will prompt the user if he wants to 
repeat the experiment with other hyperparameters, show loss graphs of all the experiments done, or classify the images from the test set.

## Required Libraries:
* tensorflow
* keras
* numpy
* sklearn
* matplotlib
* It is also recommended to have cuda installed for faster training

## Contributors:
1. [Vasilis Kiriakopoulos](https://github.com/MediaBilly)
2. [Dimitris Koutsakis](https://github.com/koutsd)
