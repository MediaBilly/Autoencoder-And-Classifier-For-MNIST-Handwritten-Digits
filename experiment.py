import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, params, history):
        self.params = params
        self.history = history
        
    
    def plot(self):        
        plt.plot(self.history['loss'], label='training data')
        plt.plot(self.history['val_loss'], label='validation data')

        title = ""
        for item in self.params:
            title += item + ": " + str(self.params[item]) + "\n"

        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc="upper right")          
        plt.title(title)
        
        plt.show()
