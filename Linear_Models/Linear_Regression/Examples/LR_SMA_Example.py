import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np


# Data Fetching and Preparation
# Download historical data for the S&P 500 index from Yahoo Finance.
sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-03-25')
print(sp500.head())  # Display the first few rows


# Feature Engineering: Calculate SMAs and Add Date Features
# Calculate three Simple Moving Averages (20, 50, 100 days) and add them as new columns.
sp500['SMA20'] = sp500['Close'].rolling(window=20).mean()
sp500['SMA50'] = sp500['Close'].rolling(window=50).mean()
sp500['SMA100'] = sp500['Close'].rolling(window=100).mean()

# Remove any rows with NaN values (resulting from the moving average calculations).
sp500.dropna(inplace=True)

# Create a 'Date' column from the DataFrame index and add a 'Days' feature
# which represents the number of days since the first available date (for regression).
sp500['Date'] = sp500.index
sp500['Days'] = (sp500['Date'] - sp500['Date'].min()).dt.days


# Setup Features and Target for Modeling
# Use 'Days' and the three SMA values as features, and 'Close' price as the target.
X = sp500[['Days', 'SMA20', 'SMA50', 'SMA100']]
y = sp500['Close']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Training and Evaluation with Cross-Validation
# Initialize the Linear Regression model.
model = LinearRegression()

# Define a mean squared error scorer where lower values are better.
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Regular 5-Fold Cross-Validation with K-Fold
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
rmse_scores = np.sqrt(-cv_scores)  # Convert negative mse to positive RMSE

# TimeSeriesSplit Cross-Validation (accounts for time order)
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_time = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
rmse_scores_time = np.sqrt(-cv_scores_time)

# Print the cross-validation results for both methods.
print("TimeSeriesSplit CV RMSE Scores:", rmse_scores_time)
print(f"Mean RMSE (TimeSeriesSplit): {rmse_scores_time.mean():.2f}")
print(f"Standard Deviation (TimeSeriesSplit): {rmse_scores_time.std():.2f}\n")

print("K-Fold CV RMSE Scores:", rmse_scores)
print(f"Mean RMSE (K-Fold): {rmse_scores.mean():.2f}")
print(f"Standard Deviation (K-Fold): {rmse_scores.std():.2f}\n")

# Train the model using the training set.
model.fit(X_train, y_train)

# Predict on the testing set.
y_pred = model.predict(X_test)

# Calculate and print the RMSE on the test set.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}')


# Prediction: Forecast the Next Day's Price
# Determine the last available date and compute the next day's numeric representation.
last_date = sp500['Date'].max()
next_day = (last_date - sp500['Date'].min()).days + 1  # Increment by one day
next_date = last_date + pd.Timedelta(days=1)  # This variable can be used if needed

# Use the last available SMA values for the prediction.
future_sma20 = sp500['SMA20'].iloc[-1]
future_sma50 = sp500['SMA50'].iloc[-1]
future_sma100 = sp500['SMA100'].iloc[-1]

# Construct the feature set for the next day.
future_features = [[next_day, future_sma20, future_sma50, future_sma100]]

# Predict and print the forecasted closing price for the next day.
prediction = model.predict(future_features)
print(f"Price tomorrow: {prediction[0].item():.2f}")


# Plotting: Visualize the Data and Predictions
plt.figure(figsize=(10, 5))
plt.plot(sp500['Close'], label='Closing Price')
plt.plot(sp500['SMA20'], label='SMA 20-day')
plt.plot(sp500['SMA50'], label='SMA 50-day')
plt.plot(sp500['SMA100'], label='SMA 100-day')
plt.scatter(y_test.index, y_pred, label='Predictions', color='red', s=10)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()