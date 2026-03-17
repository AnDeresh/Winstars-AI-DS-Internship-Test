from models.rf_model import RFModel
from models.nn_model import NNModel
from models.cnn_model import CNNModel

# MNIST Classifier using different algorithms (Random Forest, Neural Network, CNN)
class MnistClassifier:
    # A unified interface for different MNIST classification algorithms
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.model = RFModel()
        elif algorithm == "nn":
            self.model = NNModel()
        elif algorithm == "cnn":
            self.model = CNNModel()
        else:
            raise ValueError("Unsupported algorithm. Choose from 'rf', 'nn', or 'cnn'.")

    # Train the model 
    def train(self, X_train, y_train):
        return self.model.train(X_train, y_train)
    
    # Predict the labels
    def predict(self, X_test):
        return self.model.predict(X_test)