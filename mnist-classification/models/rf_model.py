from sklearn.ensemble import RandomForestClassifier

from interface import MnistClassifierInterface

# Random Forest model for MNIST classification
class RFModel(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(random_state=29)

    # Train the model 
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    # Predict the labels 
    def predict(self, X):
        return self.model.predict(X)