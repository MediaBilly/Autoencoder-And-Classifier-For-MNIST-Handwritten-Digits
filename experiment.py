import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, epochs, batch_size, history):
        self.convolutional_layers = convolutional_layers
        self.convolutional_filter_size = convolutional_filter_size
        self.convolutional_filters_per_layer = convolutional_filters_per_layer
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = history
        
    
    def plot(self):        
        plt.plot(self.history['loss'], label='training data')
        plt.plot(self.history['val_loss'], label='validation data')
        
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc="upper right")          
        plt.title(
            'Convolutional Layers: ' + str(self.convolutional_layers) 
            + '\nConvolutional filter size: ' + str(self.convolutional_filter_size)
            + '\nConvolutional filters per layer: ' + str(self.convolutional_filters_per_layer)
            + '\nBatch size: ' + str(self.batch_size)
        )
        
        plt.show()
