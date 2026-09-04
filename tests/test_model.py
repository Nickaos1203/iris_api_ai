import joblib
import numpy as np

from sklearn.metrics import accuracy_score

from src.data import load_and_prepare_data


MODEL_PATH = "models/iris_model.joblib"


def test_model_exists():
    import os

    assert os.path.exists(MODEL_PATH)


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_accuracy():
    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90