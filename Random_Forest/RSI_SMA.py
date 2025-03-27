import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import ta

# Download historical S&P 500 data from Yahoo Finance
sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')

# Calculate technical indicators: Simple Moving Averages and RSI
sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()   # 20-day SMA for short-term trends
sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()   # 50-day SMA for medium-term trends
sp500['SMA100'] = sp500['Close'].rolling(window=100).mean() # 100-day SMA for long-term trends
sp500['RSI'] = ta.momentum.RSIIndicator(sp500['Close'].squeeze(), window=14).rsi()  # 14-day RSI

# Remove rows with missing values (created by technical indicator calculations)
sp500.dropna(inplace=True)

# Remove outliers using the z-score for selected features
features_to_check = ['SMA20', 'SMA50', 'SMA100', 'RSI', 'Close']
z_scores = np.abs(zscore(sp500[features_to_check]))
threshold = 3  # Remove points that are more than 3 standard deviations away
sp500 = sp500[(z_scores < threshold).all(axis=1)]

# Add columns for date and the number of days since the start
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days

# Define the features (X) and the target variable (y)
X = sp500[['Days', 'SMA20', 'SMA50', 'SMA100', 'RSI']]
y = sp500['Close'].values.ravel() 

# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets while preserving the date order
indices = sp500.index
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, indices, test_size=0.2, random_state=42
)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform regular k-fold cross-validation and compute RMSE scores
scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scorer)
rmse_scores = np.sqrt(-cv_scores)

# Use TimeSeriesSplit to respect the temporal order in the data
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_time = cross_val_score(model, X_scaled, y, cv=tscv, scoring=scorer)
rmse_scores_time = np.sqrt(-cv_scores_time)

print("TimeSeriesSplit CV RMSE Scores:", rmse_scores_time)
print(f"Mean RMSE (TimeSeriesSplit): {rmse_scores_time.mean():.2f}")
print(f"Standard Deviation (TimeSeriesSplit): {rmse_scores_time.std():.2f}\n")

print("Cross-Validation RMSE Scores:", rmse_scores)
print(f"Mean RMSE (K-Fold): {rmse_scores.mean():.2f}")
print(f"Standard Deviation (K-Fold): {rmse_scores.std():.2f}\n")

# Train the model using the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
epsilon = 1e-10  # small constant to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%\n")
print("Minimum of y_test:", np.min(y_test))

# Predict the closing price for the next day
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1

future_features = [[
    next_day,
    sp500['SMA20'].iloc[-1],
    sp500['SMA50'].iloc[-1],
    sp500['SMA100'].iloc[-1],
    sp500['RSI'].iloc[-1]
]]
prediction = model.predict(future_features)
print(f"Predicted Price for Tomorrow: {prediction[0]:.2f}")

# Plot the actual closing prices, moving averages, and model predictions
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Close')
plt.plot(sp500['SMA20'], label='SMA 20 Days')
plt.plot(sp500['SMA50'], label='SMA 50 Days')
plt.plot(sp500['SMA100'], label='SMA 100 Days')
plt.scatter(idx_test, y_pred, label='Predictions', color='red', s=10)
plt.title('S&P 500')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
