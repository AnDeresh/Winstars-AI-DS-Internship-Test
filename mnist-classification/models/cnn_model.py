from tensorflow import keras
import numpy as np

from interface import MnistClassifierInterface

# Convolutional Neural Network model for MNIST classification
class CNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), # (26, 26, 32)
            keras.layers.MaxPooling2D(pool_size=(2, 2)), # (13, 13, 32)
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # (11, 11, 64)
            keras.layers.MaxPooling2D(pool_size=(2, 2)), # (5, 5, 64)
            keras.layers.Flatten(), # (1600)
            keras.layers.Dense(128, activation='relu'), # (128)
            keras.layers.Dense(10, activation='softmax') # (10)
        ])

    # Train the model
    def train(self, X_train, y_train):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=5)
        return self
    
    # Predict the labels
    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test.reshape(-1, 28, 28, 1)), axis=1)