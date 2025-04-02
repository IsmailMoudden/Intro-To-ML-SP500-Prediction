"""
Advanced Random Forest implementation for S&P 500 prediction with hyperparameter tuning.
This example demonstrates more sophisticated techniques for model optimization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import ta
import time

# Download and process data
def prepare_data():
    print("Downloading S&P 500 data...")
    sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
    
    print("Adding technical indicators...")
    # Moving averages
    sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
    sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()
    sp500['SMA100'] = sp500['Close'].rolling(window=100).mean()
    
    # RSI and MACD
    sp500['RSI'] = ta.momentum.RSIIndicator(sp500['Close'].squeeze(), window=14).rsi()
    macd = ta.trend.MACD(sp500['Close'])
    sp500['MACD'] = macd.macd()
    sp500['MACD_Signal'] = macd.macd_signal()
    sp500['MACD_Diff'] = macd.macd_diff()
    
    # Volatility
    sp500['Volatility'] = sp500['Close'].rolling(window=20).std()
    
    # Clean the data
    sp500.dropna(inplace=True)
    
    # Date features
    sp500['Date'] = sp500.index
    sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days
    
    return sp500

# Tune hyperparameters
def tune_hyperparameters(X, y):
    print("Tuning hyperparameters...")
    start_time = time.time()
    
    # Create parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create and fit grid search
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.2f}")
    print(f"Time taken: {(time.time() - start_time):.2f} seconds")
    
    return grid_search.best_estimator_

# Feature importance analysis
def analyze_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    
    print("\nFeature Ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {features[idx]} ({importances[idx]:.4f})")

def main():
    # Prepare data
    sp500 = prepare_data()
    
    # Define features and target
    features = ['Days', 'SMA20', 'SMA50', 'SMA100', 'RSI', 
                'MACD', 'MACD_Signal', 'MACD_Diff', 'Volatility']
    X = sp500[features]
    y = sp500['Close']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False  # Keep chronological order
    )
    
    # Tune and train model
    best_model = tune_hyperparameters(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nFinal Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Analyze feature importance
    analyze_feature_importance(best_model, features)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(sp500.index[-len(y_test):], y_test.values, label='Actual')
    plt.plot(sp500.index[-len(y_test):], y_pred, label='Predicted', alpha=0.7)
    plt.title('S&P 500 Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Predict next day
    latest_features = np.array([
        sp500['Days'].iloc[-1] + 1,  # Next day
        sp500['SMA20'].iloc[-1],
        sp500['SMA50'].iloc[-1],
        sp500['SMA100'].iloc[-1],
        sp500['RSI'].iloc[-1],
        sp500['MACD'].iloc[-1],
        sp500['MACD_Signal'].iloc[-1],
        sp500['MACD_Diff'].iloc[-1],
        sp500['Volatility'].iloc[-1]
    ]).reshape(1, -1)
    
    latest_scaled = scaler.transform(latest_features)
    prediction = best_model.predict(latest_scaled)[0]
    
    print(f"\nPredicted Price for Tomorrow: {prediction:.2f}")

if __name__ == "__main__":
    main()