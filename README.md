# ML Model for S&P 500 Price Prediction

This project demonstrates simple machine learning approaches to predict the closing price of the S&P 500 index. It serves as an educational exploration of basic data science concepts, with implementations that are deliberately straightforward to highlight fundamental principles.

## Project Overview

The implementation focuses on two prediction models - Linear Regression and Random Forest - which are applied to historical S&P 500 data enhanced with technical indicators. All concepts used in this project are documented in the `Notions` directory for reference.

## Key Concepts

### Data Collection and Preprocessing
Financial data is retrieved using the **yfinance** library. The preprocessing workflow includes:
- Loading historical S&P 500 data
- Creating technical indicators
- Managing missing values
- Detecting and filtering outliers
- Date-based feature engineering

For detailed information, see [`Notions/Data_Handeling.markdown`](Notions/Data_Handeling.markdown).

### Technical Indicators
We utilize several technical indicators to enhance the dataset:
- **SMA** (Simple Moving Averages): SMA20, SMA50, and SMA100 for short, medium, and long-term trends
- **RSI** (Relative Strength Index): Measuring price movement speed and magnitude
- **MACD** (Moving Average Convergence Divergence): Optional in some implementations

Learn more about these indicators in [`Notions/technical_indicator.markdown`](Notions/technical%20_indicator.markdown).

### Model Validation
Both models utilize:
- Train-test splitting with consideration for temporal sequence
- Cross-validation with both K-Fold and TimeSeriesSplit approaches
- Multiple performance metrics (RMSE, MAE, MAPE)

More details about validation methods in [`Notions/Performance_evaluation.markdown`](Notions/Performance_evaluation.markdown).

## Models Implemented

### 1. Linear Regression
**Directory**: [`Linear_Models/Linear_Regression/`](Linear_Models/Linear_Regression/)

- **RSI_SMA.py**: Implements linear regression with multiple features (SMA20/50/100, RSI, Days)
- **SMA.py**: Simpler implementation using only SMA features

Linear regression establishes a direct linear relationship between features and the target price. While simple, it provides a useful baseline and offers high interpretability.

Learn more about linear regression in [`Notions/Models/Linear_Regression`](Notions/Models/Linear_Regression).

### 2. Random Forest
**Directory**: [`Random_Forest/`](Random_Forest/)

- **RSI_SMA.py**: Random forest implementation with multiple features
- **SMA.py**: Implementation with only SMA features

Random Forest captures more complex relationships in the data through an ensemble of decision trees. This typically provides better predictive performance at the cost of reduced interpretability.

More details about random forest in [`Notions/Models/Random_Forest`](Notions/Models/Random_Forest).

## Performance Evaluation

Both models are evaluated using:
- **RMSE** (Root Mean Squared Error): Indicates the average prediction error in index points
- **Cross-validation**: Used to assess generalizability across different time periods
- **MAE** and **MAPE**: Additional metrics for comprehensive evaluation

## Getting Started

To run these models:

1. Ensure you have Python 3.x and the necessary libraries installed (pandas, numpy, scikit-learn, matplotlib, yfinance, ta)
2. Clone this repository
3. Run the model scripts in either the Linear_Models or Random_Forest directories

## Further Experimentation

These models are intentionally basic to highlight fundamental principles. Consider experimenting with:
- Additional technical indicators
- Feature selection techniques
- Hyperparameter optimization
- More sophisticated models like Gradient Boosting or Neural Networks

## About the S&P 500

For information about what the S&P 500 is and how to interpret predictions, see [`Notions/About_S&P500.markdown`](Notions/About_S&P500.markdown).
