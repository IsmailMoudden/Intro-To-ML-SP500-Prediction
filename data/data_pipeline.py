"""
Data Pipeline for S&P 500 Prediction Project

This module handles all data operations including:
- Data downloading from Yahoo Finance
- Technical indicator calculation
- Data preprocessing and cleaning
- Feature engineering
- Data validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SP500DataPipeline:
    """
    Complete data pipeline for S&P 500 data processing
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Default parameters
        self.symbol = "^GSPC"
        self.start_date = "2010-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
    def download_data(self, 
                     symbol: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     force_download: bool = False) -> pd.DataFrame:
        """
        Download S&P 500 data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (default: ^GSPC for S&P 500)
            start_date: Start date for data download
            end_date: End date for data download
            force_download: Force re-download even if file exists
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol or self.symbol
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        # Check if data already exists
        cache_file = self.raw_dir / f"{symbol}_{start_date}_{end_date}.pkl"
        
        if not force_download and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data downloaded for {symbol}")
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Successfully downloaded {len(data)} rows of data")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Calculating technical indicators")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Moving Average Crossovers
        df['SMA_20_50_ratio'] = df['SMA_20'] / df['SMA_50']
        df['SMA_50_200_ratio'] = df['SMA_50'] / df['SMA_200']
        
        # Volatility indicators
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_50'] = df['Returns'].rolling(window=50).std()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Support and Resistance levels (simplified)
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        
        logger.info(f"Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} technical indicators")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with target variables added
        """
        logger.info("Creating target variables")
        
        df = df.copy()
        
        # Binary classification targets
        for days in [1, 5, 10, 20, 63]:  # 1 day, 1 week, 2 weeks, 1 month, 3 months
            df[f'Target_{days}d'] = (df['Close'].shift(-days) > df['Close']).astype(int)
            df[f'Return_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
        
        # Multi-class targets (strong up, up, down, strong down)
        for days in [5, 20]:
            returns = df[f'Return_{days}d']
            df[f'Target_{days}d_Multi'] = pd.cut(
                returns, 
                bins=[-np.inf, -0.05, 0, 0.05, np.inf],
                labels=[0, 1, 2, 3]  # Strong down, down, up, strong up
            )
        
        # Volatility targets
        df['Volatility_Target_20d'] = (df['Volatility_20'].shift(-20) > df['Volatility_20']).astype(int)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame with missing values
            method: Method to handle missing values ('drop', 'forward_fill', 'interpolate')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using method: {method}")
        
        initial_rows = len(df)
        
        if method == 'drop':
            df = df.dropna()
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        
        final_rows = len(df)
        logger.info(f"Removed {initial_rows - final_rows} rows with missing values")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns
        
        Args:
            df: DataFrame to clean
            columns: List of columns to check for outliers
            method: Method to detect outliers ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers using {method} method with threshold {threshold}")
        
        initial_rows = len(df)
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                if method == 'zscore':
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < threshold]
                elif method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        final_rows = len(df_clean)
        logger.info(f"Removed {initial_rows - final_rows} outlier rows")
        
        return df_clean
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: DataFrame with features
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing features using {method} method")
        
        df_norm = df.copy()
        
        # Select numerical columns (exclude target variables and dates)
        numerical_cols = df_norm.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numerical_cols if 'Target' in col or 'Return' in col]
        feature_cols = [col for col in numerical_cols if col not in target_cols]
        
        if method == 'standard':
            for col in feature_cols:
                if df_norm[col].std() > 0:
                    df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
        elif method == 'minmax':
            for col in feature_cols:
                if df_norm[col].max() != df_norm[col].min():
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'robust':
            for col in feature_cols:
                Q1 = df_norm[col].quantile(0.25)
                Q3 = df_norm[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    df_norm[col] = (df_norm[col] - df_norm[col].median()) / IQR
        
        logger.info(f"Normalized {len(feature_cols)} feature columns")
        return df_norm
    
    def split_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, 
                   validation_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame to split
            target_col: Name of target column
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # Remove rows where target is missing
        df_clean = df.dropna(subset=[target_col])
        
        # Sort by date to maintain temporal order
        if 'Date' in df_clean.columns:
            df_clean = df_clean.sort_values('Date')
        
        # Calculate split indices
        total_size = len(df_clean)
        test_end = int(total_size * (1 - test_size))
        val_end = int(test_end * (1 - validation_size))
        
        train_df = df_clean.iloc[:val_end].copy()
        val_df = df_clean.iloc[val_end:test_end].copy()
        test_df = df_clean.iloc[test_end:].copy()
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to file
        
        Args:
            df: DataFrame to save
            filename: Name of the file
        """
        filepath = self.processed_dir / filename
        
        if filename.endswith('.csv'):
            df.to_csv(filepath, index=True)
        elif filename.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(df, f)
        elif filename.endswith('.parquet'):
            df.to_parquet(filepath, index=True)
        
        logger.info(f"Saved processed data to {filepath}")
    
    def run_full_pipeline(self, 
                          symbol: str = "^GSPC",
                          start_date: str = "2010-01-01",
                          end_date: Optional[str] = None,
                          save_intermediate: bool = True) -> pd.DataFrame:
        """
        Run the complete data pipeline
        
        Args:
            symbol: Stock symbol to download
            start_date: Start date for data
            end_date: End date for data
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Fully processed DataFrame
        """
        logger.info("Starting full data pipeline")
        
        # Download data
        df = self.download_data(symbol, start_date, end_date)
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create target variables
        df = self.create_target_variables(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, method='drop')
        
        # Remove outliers from key columns
        outlier_cols = ['Returns', 'Volatility_20', 'RSI', 'MACD']
        df = self.remove_outliers(df, outlier_cols, method='zscore', threshold=3.0)
        
        # Normalize features
        df = self.normalize_features(df, method='standard')
        
        if save_intermediate:
            self.save_processed_data(df, f"{symbol}_processed.pkl")
        
        logger.info("Data pipeline completed successfully")
        return df

def main():
    """Example usage of the data pipeline"""
    pipeline = SP500DataPipeline()
    
    # Run full pipeline
    df = pipeline.run_full_pipeline(
        symbol="^GSPC",
        start_date="2020-01-01",
        end_date="2024-01-01"
    )
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Split data
    train_df, val_df, test_df = pipeline.split_data(df, 'Target_5d')
    
    # Save split datasets
    pipeline.save_processed_data(train_df, "train_data.pkl")
    pipeline.save_processed_data(val_df, "validation_data.pkl")
    pipeline.save_processed_data(test_df, "test_data.pkl")

if __name__ == "__main__":
    main()
