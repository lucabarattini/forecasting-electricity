import os
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def apply_profile_clustering(train_df, test_df, n_clusters=5, plot=False):
    """
    Identifies consumption shapes (profiles) based ONLY on training data and 
    maps them to the test set using ClientID.
    """
    print(f"Calculating shape clusters (k={n_clusters}) based on Training data...")
    
    # Feature extraction per client
    profiles = train_df.groupby(['ClientID', 'Hour'], observed=True)['Consumption'].mean().unstack().fillna(0)

    # Scale each client (row) individually
    scaler = MinMaxScaler()
    profiles_scaled = scaler.fit_transform(profiles.T).T

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_map = pd.Series(kmeans.fit_predict(profiles_scaled), index=profiles.index)
    
    # Apply mapping
    train_df['Cluster'] = train_df['ClientID'].map(cluster_map).astype('category')
    test_df['Cluster'] = test_df['ClientID'].map(cluster_map).astype('category')

    if plot:
        print(f"Executing final K-Means with k={n_clusters} clusters...")
        cluster_labels = kmeans.labels_
        
        plt.figure(figsize=(12, 6))
        for i in range(n_clusters):
            cluster_mean = profiles_scaled[cluster_labels == i].mean(axis=0)
            cluster_size = sum(cluster_labels == i)
            plt.plot(range(1, 25), cluster_mean, label=f'Cluster {i} (n={cluster_size})', linewidth=2)

        plt.title('Normalized Daily Load Profiles by Customer Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Normalized Average kW (0 to 1)', fontsize=12)
        plt.xticks(range(1, 25))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return train_df, test_df