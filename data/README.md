# 📊 Data Directory

This directory contains all data-related functionality for the S&P 500 Prediction project.

## 📁 Directory Structure

```
data/
├── __init__.py              # Package initialization
├── data_pipeline.py         # Main data processing pipeline
├── raw/                     # Raw downloaded data (cached)
├── processed/               # Processed and cleaned data
├── external/                # External data sources (future use)
└── README.md               # This file
```

## 🚀 Data Pipeline

The `SP500DataPipeline` class provides a comprehensive solution for:

### Data Download
- **Source**: Yahoo Finance (^GSPC)
- **Caching**: Automatic caching to avoid re-downloading
- **Date Range**: Configurable start/end dates

### Technical Indicators
- **Moving Averages**: SMA (5, 10, 20, 50, 100, 200), EMA variants
- **Momentum**: RSI, MACD, Momentum indicators
- **Volatility**: Rolling standard deviation, Bollinger Bands
- **Volume**: Volume ratios, volume moving averages
- **Support/Resistance**: Dynamic levels based on rolling windows

### Target Variables
- **Binary Classification**: Price direction (up/down) for various horizons
- **Multi-class**: Strong up, up, down, strong down
- **Regression**: Actual returns for different time periods
- **Volatility**: Future volatility predictions

### Data Preprocessing
- **Missing Values**: Multiple handling strategies (drop, forward fill, interpolate)
- **Outlier Removal**: Z-score and IQR methods
- **Normalization**: Standard, min-max, and robust scaling
- **Feature Engineering**: Lag features, rolling statistics, interactions

## 📊 Data Structure

### Input Data (OHLCV)
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume

### Output Features
- **Technical Indicators**: 50+ calculated features
- **Target Variables**: Multiple prediction horizons
- **Engineered Features**: Lag, rolling, and interaction features

## 🔧 Usage Examples

### Basic Usage
```python
from data.data_pipeline import SP500DataPipeline

# Initialize pipeline
pipeline = SP500DataPipeline()

# Run full pipeline
df = pipeline.run_full_pipeline(
    symbol="^GSPC",
    start_date="2020-01-01",
    end_date="2024-01-01"
)

# Split data for modeling
train_df, val_df, test_df = pipeline.split_data(df, 'Target_5d')
```

### Custom Configuration
```python
# Download specific data
df = pipeline.download_data(
    symbol="^GSPC",
    start_date="2015-01-01",
    force_download=True
)

# Calculate indicators only
df_with_indicators = pipeline.calculate_technical_indicators(df)

# Handle missing values
df_clean = pipeline.handle_missing_values(df, method='forward_fill')
```

## 📈 Data Quality

### Validation
- **Completeness**: Automatic handling of missing values
- **Consistency**: Temporal order maintenance
- **Accuracy**: Outlier detection and removal
- **Timeliness**: Real-time data updates

### Caching
- **Raw Data**: Downloaded data cached in pickle format
- **Processed Data**: Final datasets saved for reuse
- **Efficiency**: Avoids redundant downloads and calculations

## 🔍 Monitoring

### Logging
- **Download Progress**: Data retrieval status
- **Processing Steps**: Technical indicator calculations
- **Quality Metrics**: Missing values, outliers, data shape
- **Performance**: Processing time and memory usage

### Error Handling
- **Network Issues**: Automatic retry mechanisms
- **Data Validation**: Format and content verification
- **Graceful Degradation**: Partial data processing when possible

## 🚧 Future Enhancements

### Planned Features
- **Alternative Data**: News sentiment, economic indicators
- **Real-time Updates**: Streaming data integration
- **Multi-asset Support**: Beyond S&P 500
- **Advanced Features**: Fourier transforms, wavelet analysis
- **Data Validation**: Schema validation, anomaly detection

### Integration
- **APIs**: Alpha Vantage, Quandl, FRED
- **Databases**: PostgreSQL, MongoDB
- **Cloud Storage**: AWS S3, Google Cloud Storage
- **Streaming**: Kafka, Redis

## 📚 Dependencies

### Required Packages
- `yfinance`: Yahoo Finance data access
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `ta`: Technical analysis library
- `scikit-learn`: Machine learning utilities

### Optional Packages
- `plotly`: Interactive visualizations
- `seaborn`: Statistical plotting
- `matplotlib`: Basic plotting

## ⚠️ Important Notes

1. **Educational Purpose**: This project is for learning, not trading
2. **Data Limitations**: Historical data may not predict future performance
3. **Market Hours**: Data reflects US market trading hours
4. **Holidays**: Trading holidays may affect data availability
5. **Rate Limits**: Yahoo Finance has API rate limits

## 🤝 Contributing

To contribute to the data pipeline:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** with sample data
5. **Submit** a pull request

## 📞 Support

For questions or issues:
- **Email**: ismail.moudden1@gmail.com
- **Issues**: GitHub issue tracker
- **Documentation**: Check the main README.md

---

*This data pipeline is designed to provide clean, reliable data for machine learning models while maintaining educational value and transparency.*
