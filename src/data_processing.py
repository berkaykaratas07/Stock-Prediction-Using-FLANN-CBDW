import yfinance as yf
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import numpy as np


def fetch_data(ticks, days=435, interval='60m'):
    """
    Fetches historical data for specified stock tickers using the yfinance library.

    Parameters:
        ticks (list): List of stock ticker symbols (e.g., ["AAPL", "GOOG"]).
        days (int): Number of days of historical data to retrieve.
        interval (str): Data interval (e.g., "60m" for hourly data).

    Returns:
        DataFrame: Combined historical data for all specified tickers, with 'Close'
                   and 'Volume' columns converted to float64 type.
    """
    df_list = []
    for tick in ticks:
        # Download historical data for each ticker
        df = yf.download(tick, start=pd.to_datetime('today') - timedelta(days), interval=interval)
        df['Ticker'] = tick  # Label each ticker
        df_list.append(df)

    # Combine data for all tickers
    df = pd.concat(df_list, axis=0)

    # Ensure 'Close' and 'Volume' columns are of type float64
    df['Close'] = df['Close'].astype('float64')
    df['Volume'] = df['Volume'].astype('float64')

    return df


def calculate_indicators(df):
    """
    Calculates technical indicators for the given DataFrame, including:
    - RSI (Relative Strength Index)
    - Short and Long Moving Averages for Volume
    - MAVI (Moving Average Volume Indicator)

    Parameters:
        df (DataFrame): Historical stock data with 'Close' and 'Volume' columns.

    Returns:
        DataFrame: The input DataFrame with additional columns for each indicator.
    """
    # Ensure 'Close' and 'Volume' columns are converted to numpy float64 arrays
    close_values = np.array(df['Close'], dtype='float64')
    volume_values = np.array(df['Volume'], dtype='float64')

    # Calculate RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(close_values.flatten(), timeperiod=14)

    # Calculate short and long moving averages for Volume
    df['Short_MA'] = talib.SMA(volume_values.flatten(), timeperiod=5)
    df['Long_MA'] = talib.SMA(volume_values.flatten(), timeperiod=20)

    # Calculate MAVI as the ratio of Short MA to Long MA
    df['MAVI'] = df['Short_MA'] / df['Long_MA']

    return df


def fill_missing_data(df):
    """
    Fills missing values in the DataFrame using interpolation and forward/backward filling.

    Parameters:
        df (DataFrame): DataFrame with potential missing values.

    Returns:
        DataFrame: DataFrame with missing values filled.
    """
    # Fill missing values using linear interpolation
    df.interpolate(method='linear', inplace=True)

    # Fill any remaining missing values with forward and backward filling
    df.ffill(inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill

    return df


def scale_data(X_train, X_test, y_train, y_test):
    """
    Scales the training and testing data using MinMaxScaler.

    Parameters:
        X_train (DataFrame): Training set of independent variables.
        X_test (DataFrame): Testing set of independent variables.
        y_train (Series): Training set of dependent variable.
        y_test (Series): Testing set of dependent variable.

    Returns:
        tuple: Scaled X_train, X_test, y_train, y_test, and fitted scalers for X and y.
    """
    # Initialize MinMaxScalers for both independent (X) and dependent (y) variables
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Scale training data
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # Scale testing data based on training data scales
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y
