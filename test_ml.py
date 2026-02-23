import pytest
# TODO: add necessary import
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, inference, compute_model_metrics


# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_model():
    """
    Test that train_model returns a trained LogisticRegression model.
    """
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_returns_correct_length():
    """
    Test that inference returns predictions of the correct length.
    """
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(X)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_range():
    """
    Test that compute_model_metrics returns values between 0 and 1.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1