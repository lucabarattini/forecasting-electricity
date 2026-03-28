import os
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jenkspy import jenks_breaks

def apply_volume_clustering(train_df, test_df, plot=False):
    """
    Categorizes clients by total volume (Light, Medium, Heavy) using Jenks Natural Breaks
    on the training set to prevent data leakage.
    """
    print(f"Calculating volume categorizations (Light, Medium, Heavy)...")
    
    # Calculate historical mean consumption per client.
    client_means = train_df.groupby('ClientID', observed=True)['Consumption'].mean()

    # Normalize the variance before passing it to the Jenks algorithm
    log_means = np.log1p(client_means)

    # Calculate Jenks Natural Breaks
    breaks    = jenks_breaks(log_means.values, n_classes=3)
    light_threshold_log = breaks[1]
    heavy_threshold_log = breaks[2]
    
    light_threshold = np.expm1(light_threshold_log)
    heavy_threshold = np.expm1(heavy_threshold_log)

    print(f"Jenks Breakpoints Detected:")
    print(f" > Light consumers:  below {light_threshold:.2f} kW average")
    print(f" > Medium consumers: {light_threshold:.2f} to {heavy_threshold:.2f} kW average")
    print(f" > Heavy consumers:  above {heavy_threshold:.2f} kW average")

    def categorize_client(mean_val):
        log_val = np.log1p(mean_val)
        if log_val <= light_threshold_log:
            return 'Light'
        elif log_val <= heavy_threshold_log:
            return 'Medium'
        else:
            return 'Heavy'

    # Apply mapping
    category_map = client_means.apply(categorize_client)
    train_df['Consumer_Category'] = train_df['ClientID'].map(category_map).astype('category')
    test_df['Consumer_Category'] = test_df['ClientID'].map(category_map).astype('category')

    if plot:
        client_categories = category_map.rename('Consumer_Category')
        plot_df = client_means.reset_index()
        plot_df.columns = ['ClientID', 'MeanConsumption']
        plot_df = plot_df.merge(client_categories.reset_index(), on='ClientID').sort_values('MeanConsumption')

        pcts = plot_df['Consumer_Category'].value_counts(normalize=True) * 100
        colors = {'Light': '#4CAF93', 'Medium': '#F0A500', 'Heavy': '#E05C5C'}

        # Visualization using a log-scale Historgram to map the distribution spread
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Consumer Volumetric Segments — Jenks Breakpoints (Log Scale)', fontsize=14, fontweight='bold')

        for label, color in colors.items():
            vals = np.log1p(plot_df.loc[plot_df['Consumer_Category'] == label, 'MeanConsumption'])
            ax.hist(vals, bins=60, color=color, alpha=0.65, label=f"{label} ({pcts[label]:.1f}%)", edgecolor='white', linewidth=0.4)

        ax.axvline(light_threshold_log, color='black', linestyle='--', linewidth=1.2, label=f'Light/Medium edge: {light_threshold:.0f} kW')
        ax.axvline(heavy_threshold_log, color='black', linestyle=':', linewidth=1.2, label=f'Medium/Heavy edge: {heavy_threshold:.0f} kW')

        tick_vals_kw  = [0, 10, 100, 500, 1000, 5000, 10000]
        ax.set_xticks(np.log1p(tick_vals_kw))
        ax.set_xticklabels([str(v) for v in tick_vals_kw])
        ax.set_xlim(np.log1p(0), np.log1p(10000))

        ax.set_xlabel('Historical Mean Consumption (kW)')
        ax.set_ylabel('Density Count')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.show()

    return train_df, test_df