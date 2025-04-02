import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import ta

# Download S&P 500 data
sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
print(sp500.head())

# For this purposes, we only use RSI and MACD indicator.
# Compute RSI using the ta library.
sp500['RSI'] = ta.momentum.RSIIndicator(sp500['Close'], window=14).rsi()

# Compute MACD and extract its difference.
macd = ta.trend.MACD(sp500['Close'], window_slow=26, window_fast=12, window_sign=9)
sp500['MACD_Diff'] = macd.macd_diff()

sp500.dropna(inplace=True)

# Create a time feature: number of days since the first observation.
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days

# Define features (Days, RSI, MACD_Diff) and target (Close)
X = sp500[['Days', 'RSI', 'MACD_Diff']]
y = sp500['Close']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and evaluate the linear regression model.
model = LinearRegression()
scorer = make_scorer(mean_squared_error, greater_is_better=False)
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

# Fit the model on training data and predict on the test set.
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.2f}")

# Predict the next day's price using the latest indicators.
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1
future_features = [[next_day, sp500['RSI'].iloc[-1], sp500['MACD_Diff'].iloc[-1]]]
future_prediction = model.predict(future_features)
print(f"Predicted Price for Tomorrow: {future_prediction[0]:.2f}")

# Plot the actual closing price and the predictions.
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing Price')
plt.scatter(y_test.index, y_pred, label='Predictions', color='red', s=10)
plt.title('S&P 500 Price and Predictions (RSI & MACD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
