import numpy as np
import pytest
from src.model import initialize_model, train_model, evaluate_model

@pytest.fixture
def model():
    # Initialize the model
    return initialize_model()

@pytest.fixture
def data():
    # Create mock data for training and testing
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20, 1)
    return X_train, y_train, X_test, y_test

def test_initialize_model(model):
    # Check if the model is initialized correctly
    assert model is not None

def test_train_model(model, data):
    X_train, y_train, _, _ = data
    # Train the model and check if training is successful
    trained_model = train_model(model, X_train, y_train)
    assert trained_model is not None

def test_evaluate_model(model, data):
    X_train, y_train, X_test, y_test = data
    # Train the model
    trained_model = train_model(model, X_train, y_train)

    # Evaluate the trained model
    rmse, metrics = evaluate_model(trained_model, X_test, y_test)

    # Check the correctness of evaluation metrics
    assert isinstance(rmse, float)
    assert isinstance(metrics, dict)
    assert "R2" in metrics
    assert "NSE" in metrics
    assert "MAPE" in metrics
    assert "MAE" in metrics
