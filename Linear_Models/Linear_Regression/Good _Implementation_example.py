"""
Educational Linear Regression Model using only RSI and MACD indicators for predicting S&P 500.
This script demonstrates how to download data, engineer features using technical indicators,
perform model evaluation with TimeSeriesSplit cross-validation, tune hyperparameters, and plot predictions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import time
import ta  # Added to compute technical indicators

def download_data(start_date='2010-01-01', end_date='2025-03-25', symbol='^GSPC'):
    # Download S&P 500 data from Yahoo Finance.
    data = yf.download(symbol, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def add_features(df):
    """Add RSI and MACD_Diff features for educational purposes."""
    # Compute RSI using ta library.
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    # Compute MACD difference using ta library.
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_Diff'] = macd.macd_diff()
    # Create a time feature: number of days since the first available date.
    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df.dropna(inplace=True)
    return df

def evaluate_model(model, X, y):
    # Perform TimeSeriesSplit cross-validation and report RMSE scores.
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
    rmse = np.sqrt(-cv_scores)
    print("TimeSeriesSplit CV RMSE Scores:", rmse)
    print(f"Mean RMSE: {rmse.mean():.2f}, Std: {rmse.std():.2f}\n")
    return rmse

def plot_predictions(dates, true_prices, pred_prices, title):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, true_prices, label='Actual Price', color='blue')
    plt.plot(dates, pred_prices, label='Predicted Price', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Download and preprocess the data.
    sp500 = download_data()
    sp500 = add_features(sp500)
    
    # Define features (Days, RSI, MACD_Diff) and target (Close)
    features = ['Days', 'RSI', 'MACD_Diff']
    X = sp500[features]
    y = sp500['Close']

    # Split data preserving time order (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create a pipeline that scales the data and applies linear regression.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    
    print("Evaluating the educational linear regression model with RSI and MACD:")
    evaluate_model(pipeline, X_train, y_train)
    
    # Hyperparameter tuning using GridSearchCV.
    param_grid = {'lr__fit_intercept': [True, False]}
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV RMSE: {best_rmse:.2f}")
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model on the test set.
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE using best model: {test_rmse:.2f}\n")

    # Predict the next day's closing price using the latest features.
    last_row = sp500.iloc[-1]
    next_day = float(last_row['Days'] + 1)
    future_features = np.array([[next_day, float(last_row['RSI']), float(last_row['MACD_Diff'])]])
    future_price = best_model.predict(future_features)
    print(f"Predicted S&P 500 Closing Price for Next Day: {float(future_price[0]):.2f}")
    
    # Plot the predicted prices compared to the actual prices.
    plot_predictions(X_test.index, y_test, y_pred, "S&P 500 Price Prediction - Test Set")
    

if __name__ == '__main__':
    main()
