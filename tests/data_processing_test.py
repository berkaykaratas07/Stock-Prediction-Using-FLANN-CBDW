import pandas as pd
import pytest
from src.data_processing import fetch_data, calculate_indicators, fill_missing_data, scale_data
from sklearn.model_selection import train_test_split

@pytest.fixture
def sample_df():
    # Prepare a sample dataset
    return pd.DataFrame({
        "Close": [100 + i for i in range(50)],
        "Volume": [200 + i * 10 for i in range(50)],
    })

def test_fetch_data():
    # Download data with fetch_data, checking the last 30 days
    df = fetch_data(["AAPL"], days=30, interval="1d")

    # Verify the downloaded data is not empty
    assert not df.empty
    assert "Close" in df.columns
    assert "Volume" in df.columns

def test_calculate_indicators(sample_df):
    # Calculate technical indicators
    df_with_indicators = calculate_indicators(sample_df)

    # Verify that indicator columns have been added
    assert "RSI" in df_with_indicators.columns
    assert "Short_MA" in df_with_indicators.columns
    assert "Long_MA" in df_with_indicators.columns
    assert "MAVI" in df_with_indicators.columns

def test_fill_missing_data(sample_df):
    # Add missing values and test the fill operation
    sample_df.loc[5:10, "Close"] = None
    df_filled = fill_missing_data(sample_df)

    # Verify that all missing values are filled
    assert not df_filled.isnull().values.any()

def test_scale_data(sample_df):
    # Split the data into training and testing sets
    X = sample_df[["Close", "Volume"]]
    y = sample_df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(X_train, X_test, y_train, y_test)

    # Verify the shape of scaled data matches the original shape
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert y_train_scaled.shape == (y_train.shape[0], 1)
    assert y_test_scaled.shape == (y_test.shape[0], 1)
