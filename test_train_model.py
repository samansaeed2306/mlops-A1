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
    iris = load_iris()
    X, y = iris.data, iris.target
    svm_clf = SVC(kernel='linear', C=1.0)
    svm_clf.fit(X, y)

    return svm_clf
def test_train_model_accuracy(trained_model):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = trained_model.predict(X_test)
    assert 0 <= accuracy_score(y_test, y_pred) <= 1

def test_train_model_saved(trained_model):
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(trained_model, os.path.join(model_dir, 'test_model.pkl'))
    assert os.path.exists(os.path.join(model_dir, 'test_model.pkl'))
