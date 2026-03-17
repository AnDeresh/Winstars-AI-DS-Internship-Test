from abc import ABC, abstractmethod

# Interface for the MNIST classifier
class MnistClassifierInterface(ABC):

    # Gets a set of images and their labels, trains the model, and returns itself
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    # Gets a set of images, returns a list of predicted numbers
    @abstractmethod
    def predict(self, X):
        pass