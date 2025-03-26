import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import ta as ta


sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
print(sp500.head())

# SMA
sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()
sp500['SMA100'] = sp500['Close'].rolling(window=100).mean()

#RSI
sp500['RSI'] = ta.momentum.RSIIndicator(sp500['Close'].squeeze(), window=14).rsi()

# MACD
#macd = ta.trend.MACD(sp500['Close'], window_slow=26, window_fast=12, window_sign=9)
#macd_values = np.squeeze(macd.macd().values)
#sp500['MACD'] = pd.Series(macd_values, index=sp500.index)

sp500.dropna(inplace=True)

# Fdate en jour 
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days
# Features
X = sp500[['Days', 'SMA20', 'SMA50', 'SMA100', 'RSI']]#,'MACD']]
y = sp500['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrainement 
model = LinearRegression()


# cross-validation 
scorer = make_scorer(mean_squared_error, greater_is_better=False) 
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer) 

# Convert negative MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

# cross-validation results
print(f"Cross-Validation RMSE Scores: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean():.2f}")
print(f"Standard Deviation of RMSE: {rmse_scores.std():.2f}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Stats (mse )
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}') 

#prediction sur next day
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1

# Pour feature use last one 
future_sma20 = sp500['SMA20'].iloc[-1]
future_sma50 = sp500['SMA50'].iloc[-1]
future_sma100 = sp500['SMA100'].iloc[-1]
future_rsi = sp500['RSI'].iloc[-1]
#future_macd = sp500['MACD'].iloc[-1]
future_features = [[next_day, future_sma20, future_sma50, future_sma100, future_rsi]]#, future_macd]]
#Let's goooooo
prediction = model.predict(future_features)
print(f" Price tomorrow : {prediction[0].item():.2f}")

# Trace 
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing')
plt.plot(sp500['SMA20'], label='SMA 20 day')
plt.plot(sp500['SMA50'], label='SMA 50 day')
plt.plot(sp500['SMA100'], label='SMA 100 day')
plt.scatter(y_test.index, y_pred, label='Predictions', color='red',s=10)
plt.title('S&P 500')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
