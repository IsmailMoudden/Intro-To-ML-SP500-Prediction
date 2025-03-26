import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


sp500 = yf.download('^GSPC', start='2024-01-01', end='2025-03-25')

sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
sp500.dropna(inplace=True)

sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days


X = sp500[['Days', 'SMA20']]
y = sp500['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}') 


last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1
future_sma = sp500['SMA20'].iloc[-1]

future_features = [[next_day, future_sma]]
prediction = model.predict(future_features)
print(f"Prédiction pour demain : {prediction[0].item():.2f}")

# Trace
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing')
plt.plot(sp500['SMA20'], label='SMA 20 day')
plt.plot(y_test.index, y_pred, label='Predictions', color='red')
plt.title('S&P 500 - Random Forest')
plt.xlabel('Date')
plt.ylabel('Prix')
plt.legend()
plt.show()