from tensorflow import keras 

# Load and normalize the MNIST dataset
def load_data():
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, 28*28)
    X_test = (X_test / 255.0).reshape(-1, 28*28)
    return X_train, y_train, X_test, y_test

