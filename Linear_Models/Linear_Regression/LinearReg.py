import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


sp500 = yf.download('^GSPC', start='2024-01-01', end='2025-03-25')
print(sp500.head())
# Moyenne mob 
sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()
sp500['SMA100'] = sp500['Close'].rolling(window=100).mean()
sp500.dropna(inplace=True)
# Fdate en jour 
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days
# Features
X = sp500[['Days', 'SMA20', 'SMA50', 'SMA100']]
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
next_date = last_date + pd.Timedelta(days=1) 
# Pour feature SMA use last one 
future_sma20 = sp500['SMA20'].iloc[-1]
future_sma50 = sp500['SMA50'].iloc[-1]
future_sma100 = sp500['SMA100'].iloc[-1]
future_features = [[next_day, future_sma20, future_sma50, future_sma100]]
#Let's goooooo
prediction = model.predict(future_features)
print(f" Price tomorrow : {prediction[0].item():.2f}")

# Trace 
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing')
plt.plot(sp500['SMA20'], label='SMA 20 day')
plt.plot(sp500['SMA50'], label='SMA 50-day')
plt.plot(sp500['SMA100'], label='SMA 100-day')
plt.scatter(next_date, prediction[-1], color='red', label='Prediction (Next Day)', zorder=2)
plt.title('S&P 500')
plt.xlabel('Date')
plt.ylabel('Point')
plt.legend()
plt.show()
