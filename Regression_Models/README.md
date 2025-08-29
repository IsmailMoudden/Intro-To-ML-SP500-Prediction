# Linear Regression Models for S&P 500 Prediction

This directory contains Linear Regression implementations for predicting S&P 500 prices using various technical indicators.

## Directory Structure

- **Implementation/** - Core implementation files
  - `LR_Technical_Indicators.py` - Complete implementation with multiple indicators
  
- **Examples/** - Example implementations demonstrating specific concepts
  - `LR_SMA_Example.py` - Simple example using only moving averages
  - `LR_RSI_MACD_Example.py` - Example using RSI and MACD indicators

## Technical Indicators Used

The main implementation utilizes:
- Simple Moving Averages (SMA20, SMA50, SMA100)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

## Models Features

- **Proper Time Series Validation**: Uses TimeSeriesSplit to prevent data leakage
- **Interpretable Results**: Linear regression provides clear coefficients to understand feature importance
- **Hyperparameter Tuning**: Grid search for optimal model parameters
- **Comprehensive Metrics**: RMSE, MAE, R² Score for thorough evaluation

## Getting Started

To run the basic example:
```bash
python Examples/LR_SMA_Example.py