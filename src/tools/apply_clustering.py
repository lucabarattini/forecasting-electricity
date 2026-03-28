import os
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from jenkspy import jenks_breaks

def apply_clustering(train_df, test_df, n_clusters=5):
    """
    Identifies consumption clusters based ONLY on training data and 
    maps them to the test set using ClientID.
    """
    print(f"Calculating clusters based on Training data...")
    
    print("Creating normalized hourly profiles (Shape Clustering)...")
    # Feature extraction per client (unstack creates columns for each hour)
    profiles = train_df.groupby(['ClientID', 'Hour'], observed=True)['Consumption'].mean().unstack().fillna(0)

    # MinMaxScaler scales features (columns) by default. We transpose the matrix (.T) to scale each client (row) individually, and then transpose it back to restore the original shape.
    scaler = MinMaxScaler()
    profiles_scaled = scaler.fit_transform(profiles.T).T

    # Fit KMeans on Normalized profiles
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_map = pd.Series(kmeans.fit_predict(profiles_scaled), index=profiles.index)
    
    # Apply mapping to both dataframes
    train_df['Cluster'] = train_df['ClientID'].map(cluster_map).astype('category')
    test_df['Cluster'] = test_df['ClientID'].map(cluster_map).astype('category')

    # Calculate historical mean consumption per client.
    # We strictly use 'df_train' to prevent Data Leakage from the test set.
    client_means = train_df.groupby('ClientID', observed=True)['Consumption'].mean()

    # Electricity consumption is heavily skewed (long-tail distribution). 
    # We apply a log1p transformation (log(1+x)) to normalize the variance before passing it to the Jenks algorithm, ensuring better break points.
    log_means = np.log1p(client_means)

    # Calculate Jenks Natural Breaks for 3 classes (Light, Medium, Heavy)
    breaks    = jenks_breaks(log_means.values, n_classes=3)
    light_threshold_log = breaks[1]
    heavy_threshold_log = breaks[2]

    # Revert the thresholds back to the original scale to interpret the thresholds in real-world units.
    light_threshold     = np.expm1(breaks[1])
    heavy_threshold     = np.expm1(breaks[2])

    print(f"Light consumers:  below {light_threshold:.2f} kW average")
    print(f"Medium consumers: {light_threshold:.2f} to {heavy_threshold:.2f} kW average")
    print(f"Heavy consumers:  above {heavy_threshold:.2f} kW average")

    def categorize_client(mean_val):
        """
        Helper function to classify clients based on the calculated log thresholds
        """
        log_val = np.log1p(mean_val)
        if log_val <= light_threshold_log:
            return 'Light'
        elif log_val <= heavy_threshold_log:
            return 'Medium'
        else:
            return 'Heavy'

    # Apply mapping to both dataframes
    category_map = client_means.apply(categorize_client)
    train_df['Consumer_Category'] = train_df['ClientID'].map(category_map).astype('category')
    test_df['Consumer_Category'] = test_df['ClientID'].map(category_map).astype('category')

    return train_df, test_df