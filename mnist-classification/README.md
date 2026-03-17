# MNIST Classification

Implementation of three MNIST digit classification models using a unified interface.

## Models
- **Random Forest** (`rf`) - sklearn-based classifier
- **Feed-Forward Neural Network** (`nn`) - Keras Dense layers
- **Convolutional Neural Network** (`cnn`) - Keras Conv2D layers

## Project Structure
```
mnist-classification/
├── models/
│   ├── rf_model.py      # Random Forest
│   ├── nn_model.py      # Neural Network
│   └── cnn_model.py     # Convolutional neural network
├── .gitignore           # gitignore file
├── data.py              # Data loading and preprocessing
├── demo.ipynb           # Demo notebook
├── interface.py         # Abstract base class
├── mnist_classifier.py  # Unified classifier interface
├── README.md            # This file
└── requirements.txt     # dependencies
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```python
from mnist_classifier import MnistClassifier

clf = MnistClassifier("cnn")  # "rf", "nn", or "cnn"
clf.train(X_train, y_train)
predictions = clf.predict(X_test)
```

## Demo
```bash
jupyter notebook demo.ipynb
```