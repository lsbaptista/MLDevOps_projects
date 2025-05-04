import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    """
    Unit test for the train_model function.

    This test verifies that the train_model function correctly trains a
    RandomForestClassifier model when provided with input features (X)
    and target labels (y).

    Test Steps:
    1. Define a small dataset with input features (X) and target labels (y).
    2. Call the train_model function with the dataset.
    3. Assert that the returned object is an instance of RandomForestClassifier.

    Raises:
        AssertionError: If the returned model is not an instance of RandomForestClassifier.
    """
    X = np.array([[1, 0], [0, 1]])
    y = np.array([0, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference():
    """
    Test the inference function to ensure it produces predictions with the same shape as the target labels.

    This test:
    - Creates a small dataset with features `X` and labels `y`.
    - Trains a model using the `train_model` function.
    - Generates predictions using the `inference` function.
    - Asserts that the shape of the predictions matches the shape of the labels.

    Raises:
        AssertionError: If the shape of the predictions does not match the shape of the labels.
    """
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert np.array_equal(preds.shape, y.shape)


def test_compute_model_metrics():
    """
    Unit test for the `compute_model_metrics` function.

    This test verifies that the precision, recall, and F1 score
    computed by the `compute_model_metrics` function are within
    the valid range [0, 1].

    Test Cases:
    - `y`: Ground truth labels as a NumPy array.
    - `preds`: Predicted labels as a NumPy array.

    Assertions:
    - Ensures that precision, recall, and F1 score are all between 0 and 1 (inclusive).
    """
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
