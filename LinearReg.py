import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


sp500 = yf.download('^GSPC', start='2025-01-01', end='2025-03-25')
print(sp500.head())
# Moyenne mob 
sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
sp500.dropna(inplace=True)
# Fdate en jour 
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days
# Features
X = sp500[['Days', 'SMA20']]
y = sp500['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrainement 
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#Stats (mse et )
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}') 

#prediction sur next day
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1

# Pour feature SMA use last one 
future_sma = sp500['SMA20'].iloc[-1]
future_features = [[next_day, future_sma]]
#Let's goooooo
prediction = model.predict(future_features)
print(f" Price tomorrow : {prediction[0].item():.2f}")

# Trace 
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing')
plt.plot(sp500['SMA20'], label='SMA 20 day')
plt.plot(y_test.index, y_pred, label='Predictions', color='red')
plt.title('S&P 500')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
