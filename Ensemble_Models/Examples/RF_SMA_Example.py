"""
Simple Random Forest model for S&P 500 prediction using only SMA indicators.
This basic example demonstrates the use of moving averages for price prediction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

# Download historical S&P 500 data
sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
print(sp500.head())

# Calculate Simple Moving Averages
sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()
sp500['SMA100'] = sp500['Close'].rolling(window=100).mean()
sp500.dropna(inplace=True)

# Add date-based features
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days

# Define features and target
X = sp500[['Days', 'SMA20', 'SMA50', 'SMA100']]
y = sp500['Close'].values.ravel()

# Split data while preserving indices
indices = sp500.index
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform cross-validation
scorer = make_scorer(mean_squared_error, greater_is_better=False) 
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer) 
rmse_scores = np.sqrt(-cv_scores)

# TimeSeriesSplit cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_time = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
rmse_scores_time = np.sqrt(-cv_scores_time)

# Print cross-validation results
print("TimeSeriesSplit CV RMSE Scores:", rmse_scores_time)
print(f"Mean RMSE (TimeSeriesSplit): {rmse_scores_time.mean():.2f}")
print(f"Standard Deviation (TimeSeriesSplit): {rmse_scores_time.std():.2f}\n")

print("K-Fold CV RMSE Scores:", rmse_scores)
print(f"Mean RMSE (K-Fold): {rmse_scores.mean():.2f}")
print(f"Standard Deviation (K-Fold): {rmse_scores.std():.2f}\n")

# Train the model and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}') 

# Predict tomorrow's price
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1
future_sma20 = sp500['SMA20'].iloc[-1]
future_sma50 = sp500['SMA50'].iloc[-1]
future_sma100 = sp500['SMA100'].iloc[-1]
future_features = [[next_day, future_sma20, future_sma50, future_sma100]]
prediction = model.predict(future_features)
print(f"Predicted Price for Tomorrow: {prediction[0]:.2f}")

# Create visualization
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Close')
plt.plot(sp500['SMA20'], label='SMA 20-day')
plt.plot(sp500['SMA50'], label='SMA 50-day')
plt.plot(sp500['SMA100'], label='SMA 100-day')
plt.scatter(idx_test, y_pred, label='Predictions', color='red', s=10)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('S&P 500 Price Prediction Using Moving Averages')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()