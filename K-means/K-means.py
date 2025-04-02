import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.decomposition import PCA
import ta
from datetime import datetime, timedelta

def download_data(symbol='^GSPC', start='2010-01-01', end='2025-03-25'):
    # Download historical S&P 500 data using yfinance
    return yf.download(symbol, start=start, end=end)

def add_technical_indicators(df):
    # Calculate simple moving averages (SMA)
    df['SMA20'] = df['Close'].rolling(window=20).mean()   # 20-day SMA
    df['SMA50'] = df['Close'].rolling(window=50).mean()   # 50-day SMA
    df['SMA100'] = df['Close'].rolling(window=100).mean()  # 100-day SMA

    # Calculate Relative Strength Index (RSI) for momentum measurement
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()

    # Compute volatility over a 20-day period via standard deviation
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Add momentum features: percentage change over 5, 10, and 20 days
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    
    # ENHANCEMENT: Add trend strength indicators
    df['SMA_ratio'] = df['SMA20'] / df['SMA50']  # Short vs medium trend
    df['RSI_momentum'] = df['RSI'] - df['RSI'].shift(5)  # RSI momentum
    
    return df

def create_target(df):
    # Create a binary target: 
    # 1 if the price after ~3 months (63 trading days) is higher, else 0.
    df['Target'] = df['Close'].shift(-63).gt(df['Close']).astype(int)
    return df

def prepare_features(df, features):
    # Remove rows with NaN values resulting from rolling windows and shifting.
    df.dropna(inplace=True)
    return df[features]

def plot_elbow(inertia, k_range, silhouette_scores=None):
    # Enhanced to include silhouette scores if provided
    plt.figure(figsize=(12, 6))
    
    if silhouette_scores is None:
        # Original elbow plot
        plt.plot(k_range, inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
    else:
        # Side-by-side plots with silhouette scores
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertia, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, marker='o')
        plt.title('Silhouette Score Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score (higher is better)')
        plt.grid(True)
        plt.tight_layout()
        
    plt.savefig('kmeans_cluster_selection.png')
    plt.close()

def optimize_clusters(X_scaled, k_range=range(2, 11)):
    """Find optimal k using both inertia and silhouette scores"""
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score for k > 1
        if k > 1:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)  # No silhouette score for k=1
    
    # Plot both metrics for cluster selection
    plot_elbow(inertia, list(k_range), silhouette_scores)
    
    # Recommend k based on silhouette score
    best_k_idx = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_k_idx]
    print(f"Recommended k based on silhouette score: {best_k}")
    
    return best_k, inertia, silhouette_scores

def plot_clusters(X_pca, df, k):
    # Visualize clusters in 2D space using the first two principal components.
    plt.figure(figsize=(12, 8))
    for cluster in range(k):
        mask = df['Cluster'] == cluster
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f'Cluster {cluster}', alpha=0.7)
    plt.title('Market Regimes Identified by K-means')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('kmeans_clusters.png')
    plt.close()

def plot_predictions(df, k):
    # Plot multiple subplots:
    # 1. S&P 500 closing price over time.
    # 2. Cluster assignment for each trading day.
    # 3. Comparison of actual vs predicted market direction.
    plt.figure(figsize=(14, 10))

    # Price Chart
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], 'b-')
    plt.title('S&P 500 Price')
    plt.grid(True)

    # Cluster Assignment Chart
    plt.subplot(3, 1, 2)
    for cluster in range(k):
        mask = df['Cluster'] == cluster
        plt.scatter(df.index[mask], np.ones(sum(mask)) * cluster,
                    c=f'C{cluster}', alpha=0.7, label=f'Cluster {cluster}')
    plt.title('Cluster Assignment Over Time')
    plt.yticks(range(k))
    plt.grid(True)

    # Prediction vs Actual Market Direction
    plt.subplot(3, 1, 3)
    plt.scatter(df.index, df['Target'], c='green', alpha=0.5,
                label='Actual (Up = 1, Down = 0)')
    plt.scatter(df.index, df['Predicted'], c='red', alpha=0.5,
                marker='x', label='Predicted')
    plt.title('Actual vs Predicted Market Direction (3 Months Ahead)')
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('kmeans_prediction.png')
    plt.show()

def improve_cluster_predictions(sp500, k):
    """Enhanced version with recency weighting and threshold adjustment"""
    # Get overall market statistics
    overall_up_ratio = sp500['Target'].mean()
    print(f"Overall market UP ratio: {overall_up_ratio:.2%}")
    
    # Initialize containers
    cluster_predictions = {}
    cluster_accuracy = {}
    cluster_confidence = {}
    
    # For each cluster, analyze predictive power with recent data emphasis
    for cluster in range(k):
        cluster_data = sp500[sp500['Cluster'] == cluster]
        
        if len(cluster_data) > 0:
            # Calculate statistics for different time periods
            all_periods_ratio = cluster_data['Target'].mean()
            
            # Calculate metrics for recent period (last 6 months if available)
            recent_cutoff = sp500.index.max() - pd.Timedelta(days=180)
            recent_data = cluster_data[cluster_data.index > recent_cutoff]
            
            # If we have enough recent data, use it with higher weight
            if len(recent_data) >= 20:  # Minimum sample size
                recent_ratio = recent_data['Target'].mean()
                # Weighted average: 70% recent, 30% all-time
                weighted_ratio = 0.7 * recent_ratio + 0.3 * all_periods_ratio
            else:
                weighted_ratio = all_periods_ratio
            
            # Adjust threshold based on overall market bias
            # Require stronger signal to go against market trend
            base_threshold = 0.5
            if overall_up_ratio > 0.6:  # Strong up market
                down_threshold = 0.45  # Need stronger signal to predict down
                up_threshold = 0.52    # Lower threshold to predict up
            elif overall_up_ratio < 0.4:  # Strong down market
                down_threshold = 0.48  # Lower threshold to predict down
                up_threshold = 0.55    # Need stronger signal to predict up
            else:  # Balanced market
                down_threshold = 0.48
                up_threshold = 0.52
                
            # Make prediction based on thresholds
            if weighted_ratio > up_threshold:
                prediction = 1  # UP
            elif weighted_ratio < down_threshold:
                prediction = 0  # DOWN
            else:
                # If in the uncertainty zone, go with the market trend
                prediction = 1 if overall_up_ratio > 0.5 else 0
                
            # Calculate confidence and expected accuracy
            confidence = abs(weighted_ratio - 0.5) * 2  # Scale to 0-1
            accuracy = weighted_ratio if prediction == 1 else (1 - weighted_ratio)
            
            # Store results
            cluster_predictions[cluster] = prediction
            cluster_accuracy[cluster] = accuracy
            cluster_confidence[cluster] = confidence
            
            # Print detailed statistics
            print(f"Cluster {cluster}: {len(cluster_data)} points")
            print(f"  - All-time UP ratio: {all_periods_ratio:.2%}")
            if len(recent_data) >= 20:
                print(f"  - Recent UP ratio: {recent_ratio:.2%} (n={len(recent_data)})")
            print(f"  - Weighted UP ratio: {weighted_ratio:.2%}")
            print(f"  - Prediction: {'UP' if prediction == 1 else 'DOWN'} " +
                  f"(confidence: {confidence:.2%})")
            print(f"  - Expected accuracy: {accuracy:.2%}")
    
    return cluster_predictions, cluster_accuracy, cluster_confidence

def add_feature_analysis(sp500, features, k):
    """Analyze how features differ across clusters"""
    # Calculate feature means by cluster
    cluster_features = sp500.groupby('Cluster')[features].mean()
    
    # Plot feature importance by cluster
    plt.figure(figsize=(14, 8))
    cluster_features.plot(kind='bar', ax=plt.gca())
    plt.title('Average Feature Values by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Standardized Value')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('cluster_features.png')
    plt.close()
    
    # Return for further analysis
    return cluster_features

def evaluate_trading_performance(sp500, holding_period=63):
    """Calculate financial performance metrics for the strategy"""
    try:
        # Clone dataframe to avoid modifying original
        eval_df = sp500.copy()
        
        # Calculate future returns over holding period
        eval_df['Future_Return'] = eval_df['Close'].pct_change(holding_period).shift(-holding_period)
        
        # Debug information
        print(f"Created Future_Return column. Sample values: {eval_df['Future_Return'].head().tolist()}")
        print(f"NaN values in Future_Return: {eval_df['Future_Return'].isna().sum()}/{len(eval_df)}")
        
        # Strategy returns - positive when prediction matches direction
        eval_df['Strategy_Return'] = np.where(
            eval_df['Predicted'] == eval_df['Target'],
            abs(eval_df['Future_Return']),  # Correct prediction = positive return
            -abs(eval_df['Future_Return'])  # Wrong prediction = negative return
        )
        
        # Remove NaN from future returns that extend beyond data
        eval_df = eval_df.loc[eval_df['Future_Return'].notna()]
        
        if len(eval_df) == 0:
            print("Warning: No valid data after removing NaN values")
            return {'win_rate': None, 'return': None}
        
        # Buy and hold returns for comparison - FIX the deprecation warnings
        first_close = eval_df['Close'].iloc[0].item()  # Use .item() to properly extract the scalar value
        last_close = eval_df['Close'].iloc[-1].item()  
        trading_days = (eval_df.index[-1] - eval_df.index[0]).days
        annual_factor = 365 / max(trading_days, 1)  # Avoid division by zero
        buy_hold_return = (last_close / first_close - 1) * annual_factor
        
        # Calculate strategy performance
        strategy_return = eval_df['Strategy_Return'].mean() * (252/holding_period)  # Annualized
        win_rate = (eval_df['Predicted'] == eval_df['Target']).mean()
        
        # Print results
        print("\nTrading Performance Metrics:")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Annualized Strategy Return: {strategy_return:.2%}")
        print(f"Buy & Hold Return (annualized): {buy_hold_return:.2%}")
        
        # Plot equity curve only if we have sufficient data
        if len(eval_df) > 1:
            plt.figure(figsize=(10, 6))
            cumulative_returns = (1 + eval_df['Strategy_Return']).cumprod()
            plt.plot(eval_df.index, cumulative_returns)
            plt.title('Strategy Equity Curve')
            plt.ylabel('Cumulative Return')
            plt.grid(True)
            plt.savefig('strategy_performance.png')
            plt.close()
        
        return {'win_rate': win_rate, 'return': strategy_return}
    
    except Exception as e:
        print(f"Error in evaluate_trading_performance: {e}")
        import traceback
        traceback.print_exc()
        return {'win_rate': None, 'return': None}

def main():
    # Step 1: Data Acquisition
    sp500 = download_data()
    print(sp500.head())  # Display first 5 rows of data

    # Step 2: Compute technical indicators and momentum features
    sp500 = add_technical_indicators(sp500)

    # Step 3: Create the target variable for predictive analysis
    sp500 = create_target(sp500)

    # Define features used for clustering (now with enhanced features)
    features = ['SMA20', 'SMA50', 'SMA100', 'RSI', 'Volatility',
                'Return_5d', 'Return_10d', 'Return_20d',
                'SMA_ratio', 'RSI_momentum']
    X = prepare_features(sp500, features)

    # Step 4: Standardize features for clustering stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Use advanced methods to select optimal number of clusters
    k_range = range(2, 11)
    k, inertia, silhouette_scores = optimize_clusters(X_scaled, k_range)
    
    # Step 6: Apply K-means clustering with selected k value.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sp500['Cluster'] = kmeans.fit_predict(X_scaled)

    # Step 7: For each cluster, assess its predictive power with enhanced method
    cluster_predictions, cluster_accuracy, cluster_confidence = improve_cluster_predictions(sp500, k)

    # Step 8: Map cluster-based predictions back to the dataset.
    sp500['Predicted'] = sp500['Cluster'].map(cluster_predictions)

    # Step 9: Evaluate classifier performance via accuracy and report.
    accuracy = accuracy_score(sp500['Target'], sp500['Predicted'])
    print(f"\nOverall accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(sp500['Target'], sp500['Predicted'], zero_division=0))
    
    # Step 9B: Evaluate trading performance
    trading_metrics = evaluate_trading_performance(sp500)

    # Step 10: Reduce dimensions with PCA for 2D visualization of clusters.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plot_clusters(X_pca, sp500, k)

    # Step 11: Visualize the evolution of the market and predictions over time.
    plot_predictions(sp500, k)

    # Step 12: Analyze feature importance across clusters.
    cluster_features = add_feature_analysis(sp500, features, k)
    print("\nCluster Feature Profiles:")
    print(cluster_features)

    # Step 13: Predict the market direction using the most recent data point.
    latest_features = sp500[features].iloc[-1:].values
    latest_scaled = scaler.transform(latest_features)
    latest_cluster = kmeans.predict(latest_scaled)[0]
    latest_prediction = cluster_predictions[latest_cluster]
    latest_confidence = cluster_confidence[latest_cluster]
    
    print(f"\nCurrent market is in Cluster {latest_cluster}")
    print(f"3-Month Prediction: {'UP' if latest_prediction == 1 else 'DOWN'}")
    print(f"Prediction confidence: {latest_confidence:.2%}")
    print(f"Historical accuracy for this cluster: {cluster_accuracy[latest_cluster]:.2%}")

if __name__ == "__main__":
    main()