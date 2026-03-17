from tensorflow import keras
import numpy as np

from interface import MnistClassifierInterface

# Neural Network model for MNIST classification
class NNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(28*28,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    # Train the model
    def train(self, X_train, y_train):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=5)
        return self
    
    # Predict the labels
    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)