"""
Random Forest model for S&P 500 prediction using technical indicators.
This implementation uses multiple technical indicators (SMA, RSI, MACD) 
along with proper time series validation techniques.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import ta

def download_data(symbol='^GSPC', start='2010-01-01', end='2025-03-25'):
    """Download historical S&P 500 data from Yahoo Finance"""
    data = yf.download(symbol, start=start, end=end)
    print(f"Downloaded {len(data)} rows of data from {start} to {end}")
    print(data.head())
    return data

def add_technical_indicators(df):
    """Add various technical indicators to the dataframe"""
    # Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()   # Short-term trend
    df['SMA50'] = df['Close'].rolling(window=50).mean()   # Medium-term trend
    df['SMA100'] = df['Close'].rolling(window=100).mean() # Long-term trend
    
    # RSI - Momentum indicator
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    
    # MACD - Trend indicator
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    return df

def clean_data(df, threshold=3):
    """Remove outliers and handle missing values"""
    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    
    # Remove outliers using z-score
    features_to_check = ['SMA20', 'SMA50', 'SMA100', 'RSI', 'Close']
    z_scores = np.abs(zscore(df[features_to_check]))
    df_clean = df[(z_scores < threshold).all(axis=1)]
    
    # Calculate removed percentage
    removed_pct = (len(df) - len(df_clean)) / len(df) * 100
    print(f"Removed {removed_pct:.2f}% of data points as outliers")
    
    # Add date features
    df_clean['Date'] = df_clean.index
    df_clean['Days'] = (df_clean['Date'] - df_clean['Date'].min()).dt.days
    
    return df_clean

def evaluate_model(model, X, y, tscv):
    """Evaluate model using both regular and time series cross-validation"""
    # Create scorer (negative MSE to maximize)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    # Regular k-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
    rmse_scores = np.sqrt(-cv_scores)
    
    # Time series cross-validation
    cv_scores_time = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
    rmse_scores_time = np.sqrt(-cv_scores_time)
    
    # Print results
    print("TimeSeriesSplit CV RMSE Scores:", rmse_scores_time)
    print(f"Mean RMSE (TimeSeriesSplit): {rmse_scores_time.mean():.2f}")
    print(f"Standard Deviation (TimeSeriesSplit): {rmse_scores_time.std():.2f}\n")
    
    print("K-Fold CV RMSE Scores:", rmse_scores)
    print(f"Mean RMSE (K-Fold): {rmse_scores.mean():.2f}")
    print(f"Standard Deviation (K-Fold): {rmse_scores.std():.2f}\n")
    
    return rmse_scores_time, rmse_scores

def calculate_metrics(y_true, y_pred):
    """Calculate and print comprehensive evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    epsilon = 1e-10  # small constant to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R² Score: {r2:.2f}\n")
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}

def predict_next_day(model, df, feature_columns, scaler=None):
    """Predict the next day's price using the trained model"""
    last_date = df['Date'].max()
    next_day = (last_date - df['Date'].min()).days + 1
    
    # Get the latest values for all features
    future_features = []
    for col in feature_columns:
        if col == 'Days':
            future_features.append(next_day)
        else:
            future_features.append(df[col].iloc[-1])
    
    # Apply same scaling if a scaler was used
    if scaler:
        future_features = scaler.transform([future_features])
    else:
        future_features = [future_features]
    
    # Make prediction
    prediction = model.predict(future_features)
    print(f"Predicted Price for Tomorrow: {prediction[0]:.2f}")
    
    return prediction[0]

def plot_results(df, y_test, y_pred, test_indices):
    """Create visualization of results"""
    plt.figure(figsize=(12, 6))
    
    # Plot price and moving averages
    plt.plot(df['Close'], label='Close Price', alpha=0.6)
    plt.plot(df['SMA20'], label='SMA 20 Days', alpha=0.6)
    plt.plot(df['SMA50'], label='SMA 50 Days', alpha=0.6)
    plt.plot(df['SMA100'], label='SMA 100 Days', alpha=0.6)
    
    # Plot test predictions
    plt.scatter(test_indices, y_pred, label='Predictions', color='red', s=10)
    
    # Add labels and legend
    plt.title('S&P 500 Price Prediction with Random Forest', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('rf_predictions.png')
    plt.show()

def feature_importance(model, feature_names):
    """Display feature importance from the Random Forest model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    plt.show()
    
    print("Feature Ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")

def main():
    # Step 1: Download data
    sp500 = download_data()
    
    # Step 2: Add technical indicators
    sp500 = add_technical_indicators(sp500)
    
    # Step 3: Clean data
    sp500 = clean_data(sp500)
    
    # Step 4: Define features and target
    feature_columns = ['Days', 'SMA20', 'SMA50', 'SMA100', 'RSI', 'MACD_Diff']
    X = sp500[feature_columns]
    y = sp500['Close'].values.ravel()
    
    # Step 5: Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 6: Split data while preserving indices
    indices = sp500.index
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, indices, test_size=0.2, random_state=42
    )
    
    # Step 7: Initialize and evaluate model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_time, rmse_kfold = evaluate_model(model, X_scaled, y, tscv)
    
    # Step 8: Train final model
    print("Training final model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Step 9: Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Step 10: Predict next day's price
    next_price = predict_next_day(model, sp500, feature_columns, scaler)
    
    # Step 11: Visualize results
    plot_results(sp500, y_test, y_pred, idx_test)
    
    # Step 12: Show feature importance
    feature_importance(model, feature_columns)
    
    return model, sp500, metrics

if __name__ == "__main__":
    main()