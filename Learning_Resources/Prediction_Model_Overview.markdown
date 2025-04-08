# Overview of Building a Prediction Model and Calculating Technical Indicators

This document provides a high-level explanation of the steps used to build the prediction model for the S&P 500 along with the rationale behind computing technical indicators such as SMAs and RSI.

---

## 1. Data Collection and Preprocessing

- **Data Retrieval:**  
  Historical data is downloaded from Yahoo Finance to ensure a reliable source.  
  - *Rationale:* Using a consistent data source helps in reproducible model training.

- **Data Cleaning:**  
  Dropping rows with missing values (`dropna()`) ensures that only complete data is used.  
  - *Rationale:* Incomplete or missing data can distort calculations and lead to inaccurate predictions.

---

## 2. Feature Engineering

### A. Time Feature
- **Conversion to 'Days':**  
  The code computes the number of days since the start date as:  
  `df['Days'] = (df['Date'] - df['Date'].min()).dt.days`  
  - *Explanation:* Subtracting the minimum date provides a numerical time feature that reflects the progression of data over time.

### B. Simple Moving Averages (SMA)
- **Calculation:**  
  SMAs are computed using a rolling window (e.g. 20, 50, 100 days).  
  - *Formula:*  
    `df['SMA20'] = df['Close'].rolling(window=20).mean()`  
  - *Rationale:* Moving averages smooth out short-term fluctuations to highlight underlying trends.

### C. Relative Strength Index (RSI)
- **Calculation:**  
  The RSI is computed with:  
  `RSI = 100 - (100 / (1 + avg_gain/avg_loss))`  
  where `avg_gain` and `avg_loss` are rolling averages of gains and losses over 14 days.
  - *Rationale:*  
    - The formula standardizes momentum into a percentage (ranging from 0 to 100).  
    - Subtracting from 100 is part of the standard RSI formula.
  
### D. Prediction Target Alignment (Time Shift and –1 usage)
- **Accessing the Latest Data:**  
  Often code uses, for example, `iloc[-1]` to get the most recent row’s data.  
  - *Explanation:* The index `-1` refers to the last element in a DataFrame, ensuring that predictions are based on the most up-to-date information.
- **Shifting Target Variables:**  
  In some scripts, a shift (e.g. `shift(-63)`) is applied to align current features with future outcomes (such as predicting 3 months ahead).  
  - *Rationale:* Shifting ensures that the target variable reflects a future value rather than the current or past value, which is critical in prediction tasks.

---

## 3. Model Building Pipeline

- **Pipeline Structure:**  
  A scikit-learn Pipeline is used to combine scaling (using StandardScaler) and the regression model.  
  - *Rationale:*  
    - Scaling ensures that features with different ranges contribute equally.
    - A pipeline improves reproducibility and simplifies cross-validation.

- **Cross-Validation:**  
  *TimeSeriesSplit* is used to maintain the chronological order, ensuring that past data is used to predict the future.  
  - *Explanation:* Standard k-fold CV might mix future data, causing data leakage.

---

## 4. Summary

This overview explains why:
- A numerical time feature is created by subtracting the minimum date.
- SMAs are computed over specified windows to smooth noise.
- RSI is calculated with its standard formula involving a division of average gains and losses and then subtracting the result from 100.
- The latest data is accessed with `-1` indexing to guarantee forecasts are made on the most recent available values.
- Shifting the target (e.g. `shift(-63)`) aligns present features with future outcomes.

These design choices are elaborated upon in the corresponding model and technical indicator files in the project. Reviewing this file should help in understanding the assumptions and calculations behind the prediction model.

---

# Understanding Model.fit() and Model.predict()

## General Concept

- **fit(X, y)**
  - The `fit` method is used to train a model on your dataset.
  - It takes in training data (features X and often target y) and adjusts the model’s internal parameters to best capture the patterns in the data.
  - For example, in regression, it calculates coefficients that minimize the error between predictions and actual values.

- **predict(X)**
  - After training, the `predict` method uses the learned parameters to make predictions on new or existing data.
  - It takes input features X and outputs predicted values (or classes, depending on the model).

## In Specific Models

### Linear Regression
- **fit:**  
  - Computes the best-fitting straight line by minimizing the Mean Squared Error (MSE) between the predicted and actual outputs.
  - Learns coefficients (slope and intercept) that define the linear relationship.
- **predict:**  
  - Uses the learned slope and intercept to calculate y-values for given x-values, following the equation: y = slope * x + intercept.

### K-means Clustering
- **fit:**  
  - Determines the positions of k cluster centroids by grouping the training data.
  - It iteratively assigns data points to the nearest centroid and then recalculates centroid positions until convergence.
- **predict:**  
  - Assigns new data points to the nearest centroid from the ones learned during the fitting process.
  - The prediction is simply the cluster label indicating the closest cluster.

In summary, regardless of the model type, `fit` is about learning from data, while `predict` applies the learned model to generate outputs.

---

*Ce document sert de guide global pour comprendre la logique et les formules utilisées dans le projet de prédiction du S&P 500.*
