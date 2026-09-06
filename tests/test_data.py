from src.pipeline_model import load_and_prepare_data

import numpy as np


def test_data_shape():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)


def test_data_classes():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    assert len(set(y_train)) == 3
    assert len(set(y_test)) == 3


def test_no_missing_values():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()