import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import ta

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

def plot_elbow(inertia, k_range):
    # 'inertia' is a list of sum-of-squared distances computed for each k value.
    # 'k_range' is a list of the respective number of clusters tested.
    # This function plots the elbow curve to help visually select the optimal number of clusters.
    # The "elbow" point marks where increasing clusters yields diminishing improvement.
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('kmeans_elbow.png')
    plt.close()

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

def main():
    # Step 1: Data Acquisition
    sp500 = download_data()
    print(sp500.head())  # Display first few rows of data

    # Step 2: Compute technical indicators and momentum features
    sp500 = add_technical_indicators(sp500)

    # Step 3: Create the target variable for predictive analysis
    sp500 = create_target(sp500)

    # Define features used for clustering
    features = ['SMA20', 'SMA50', 'SMA100', 'RSI', 'Volatility',
                'Return_5d', 'Return_10d', 'Return_20d']
    X = prepare_features(sp500, features)

    # Step 4: Standardize features for clustering stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Use the Elbow Method to select optimal number of clusters.
    inertia = []
    k_range = range(2, 11)
    for k_val in k_range:
        kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inertia.append(kmeans_temp.inertia_)
    plot_elbow(inertia, list(k_range))

    # Step 6: Apply K-means clustering with selected k value.
    k = 5  # Selected number of clusters; adjust per the elbow plot insights.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sp500['Cluster'] = kmeans.fit_predict(X_scaled)

    # Step 7: For each cluster, assess its predictive power.
    cluster_predictions = {}
    cluster_accuracy = {}
    for cluster in range(k):
        cluster_data = sp500[sp500['Cluster'] == cluster]
        if len(cluster_data) > 0:
            pct_up = cluster_data['Target'].mean()  # Proportion of upward movements
            prediction = 1 if pct_up > 0.5 else 0   # Majority class decision
            cluster_predictions[cluster] = prediction
            cluster_accuracy[cluster] = max(pct_up, 1 - pct_up)
            print(f"Cluster {cluster}: {len(cluster_data)} points, Up: {pct_up:.2%}, Prediction: {'Up' if prediction == 1 else 'Down'}")

    # Step 8: Map cluster-based predictions back to the dataset.
    sp500['Predicted'] = sp500['Cluster'].map(cluster_predictions)

    # Step 9: Evaluate classifier performance via accuracy and report.
    accuracy = accuracy_score(sp500['Target'], sp500['Predicted'])
    print(f"\nOverall accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(sp500['Target'], sp500['Predicted']))

    # Step 10: Reduce dimensions with PCA for 2D visualization of clusters.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plot_clusters(X_pca, sp500, k)

    # Step 11: Visualize the evolution of the market and predictions over time.
    plot_predictions(sp500, k)

    # Step 12: Predict the market direction using the most recent data point.
    latest_features = sp500[features].iloc[-1:].values
    latest_scaled = scaler.transform(latest_features)
    latest_cluster = kmeans.predict(latest_scaled)[0]
    latest_prediction = cluster_predictions[latest_cluster]
    print(f"\nCurrent market is in Cluster {latest_cluster}")
    print(f"3-Month Prediction: {'UP' if latest_prediction == 1 else 'DOWN'}")
    print(f"Historical accuracy for this cluster: {cluster_accuracy[latest_cluster]:.2%}")

if __name__ == "__main__":
    main()