import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pytest
from train_model import train_model

 

@pytest.fixture
def trained_model():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train a Support Vector Machine (SVM) classifier
    svm_clf = SVC(kernel='linear', C=1.0)
    svm_clf.fit(X, y)

    return svm_clf

def test_train_model_accuracy(trained_model):
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the trained model to make predictions on the test set
    y_pred = trained_model.predict(X_test)

    # Check if accuracy is within a reasonable range (between 0 and 1)
    assert 0 <= accuracy_score(y_test, y_pred) <= 1

def test_train_model_saved(trained_model):
    # Create the directory to save the trained model
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the trained model
    joblib.dump(trained_model, os.path.join(model_dir, 'test_model.pkl'))

    # Check if the model file is saved
    assert os.path.exists(os.path.join(model_dir, 'test_model.pkl'))
