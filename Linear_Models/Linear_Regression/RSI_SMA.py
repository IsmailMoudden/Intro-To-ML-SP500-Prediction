import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import ta

# Download S&P 500 data
sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
print(sp500.head())


# Technical Indicator 

sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()   
sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()    
sp500['SMA100'] = sp500['Close'].rolling(window=100).mean()  

sp500['RSI'] = ta.momentum.RSIIndicator(sp500['Close'].squeeze(), window=14).rsi()


# macd = ta.trend.MACD(sp500['Close'], window_slow=26, window_fast=12, window_sign=9)
# macd_values = np.squeeze(macd.macd().values)
# sp500['MACD'] = pd.Series(macd_values, index=sp500.index)


sp500.dropna(inplace=True)

# Convert the date
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days

# Features (X) and target (y)
X = sp500[['Days', 'SMA20', 'SMA50', 'SMA100', 'RSI']]
y = sp500['Close']

# Split Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()

# Scorer using MSE, note: greater_is_better=False to use it with cross_val_score
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform k-fold cross-validation (k=5) and compute RMSE scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
rmse_scores = np.sqrt(-cv_scores)
print(f"Cross-Validation RMSE Scores (K-Fold): {rmse_scores}")
print(f"Mean RMSE (K-Fold): {rmse_scores.mean():.2f}")
print(f"Standard Deviation (K-Fold): {rmse_scores.std():.2f}\n")


tscv = TimeSeriesSplit(n_splits=5)
cv_scores_time = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
rmse_scores_time = np.sqrt(-cv_scores_time)
print("TimeSeriesSplit CV RMSE Scores:", rmse_scores_time)
print(f"Mean RMSE (TimeSeriesSplit): {rmse_scores_time.mean():.2f}")
print(f"Standard Deviation (TimeSeriesSplit): {rmse_scores_time.std():.2f}\n")

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.2f}")

# Get the last date and calculate the number of days for the next prediction
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1

# Use the last available technical indicator values for the future prediction
future_sma20 = sp500['SMA20'].iloc[-1]
future_sma50 = sp500['SMA50'].iloc[-1]
future_sma100 = sp500['SMA100'].iloc[-1]
future_rsi = sp500['RSI'].iloc[-1]

# Feature set for the next day prediction
future_features = [[next_day, future_sma20, future_sma50, future_sma100, future_rsi]]

# Predict 
future_prediction = model.predict(future_features)
print(f"Predicted Price for Tomorrow: {future_prediction[0]:.2f}")

# Plot 
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing Price')
plt.plot(sp500['SMA20'], label='SMA 20 Days')
plt.plot(sp500['SMA50'], label='SMA 50 Days')
plt.plot(sp500['SMA100'], label='SMA 100 Days')
plt.scatter(y_test.index, y_pred, label='Predictions', color='red', s=10)
plt.title('S&P 500 Price and Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
