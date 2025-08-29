# Random Forest Models for S&P 500 Prediction

This directory contains Random Forest implementations for predicting S&P 500 prices using various technical indicators.

## Directory Structure

- **Implementation/** - Core implementation files
  - `RF_Technical_Indicators.py` - Complete implementation with multiple indicators
  
- **Examples/** - Example implementations demonstrating specific concepts
  - `RF_SMA_Example.py` - Simple example using only moving averages
  - `RF_Advanced_Example.py` - Advanced example with hyperparameter tuning

## Technical Indicators Used

The main implementation utilizes:
- Simple Moving Averages (SMA20, SMA50, SMA100)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Volatility measures

## Models Features

- **Proper Time Series Validation**: Uses TimeSeriesSplit to prevent data leakage
- **Outlier Detection**: Z-score based filtering to improve robustness
- **Feature Importance Analysis**: Visualization of most predictive features
- **Comprehensive Metrics**: RMSE, MAE, MAPE and R² Score for thorough evaluation

## Getting Started

To run the basic example:
```bash
python Examples/RF_SMA_Example.py