import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np


sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
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
y = sp500['Close'].values.ravel()

# Modified train-test split to preserve indices
indices = sp500.index
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)

# Entrainement 
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Regular K-Fold cross-validation
scorer = make_scorer(mean_squared_error, greater_is_better=False) 
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer) 
# Convert negative MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

# TimeSeriesSplit cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_time = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
rmse_scores_time = np.sqrt(-cv_scores_time)

# Print results from both cross-validation methods
print("TimeSeriesSplit CV RMSE Scores:", rmse_scores_time)
print(f"Mean RMSE (TimeSeriesSplit): {rmse_scores_time.mean():.2f}")
print(f"Standard Deviation (TimeSeriesSplit): {rmse_scores_time.std():.2f}\n")

print("K-Fold CV RMSE Scores:", rmse_scores)
print(f"Mean RMSE (K-Fold): {rmse_scores.mean():.2f}")
print(f"Standard Deviation (K-Fold): {rmse_scores.std():.2f}\n")

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
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
plt.scatter(idx_test, y_pred, label='Predictions', color='red', s=10)
plt.xlabel('Date')
plt.ylabel('Point')
plt.legend()
plt.show()