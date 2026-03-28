import os
import sys

# Allow running from anywhere by adding the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.tools import add_temporal_features, get_national_weather, clean_clients, load_raw_data, add_lags_and_rolling, apply_profile_clustering, apply_volume_clustering

def process_data(input_path, output_path):
    """
    Main pipeline to transform raw electricity data into a feature-engineered 
    dataset ready for machine learning.
    """

    # 1. Load Data
    print("Loading data...")
    df = load_raw_data(input_path)
    
    # 2. Temporal Features (on wide format to save time)
    print("Adding temporal features...")
    df = add_temporal_features(df)
    
    # 3. Melt
    print("Melting...")
    # Define the columns that should remain fixed (the time features)
    # All other columns (the 370 clients) will be melted into rows
    fixed_vars = ['Date', 'Weekday', 'Hour', 'Month', 'Is_Weekend', 'Is_Holiday']

    df_long = pd.melt(
        df, 
        id_vars=fixed_vars,             # The columns to keep fixed
        var_name='ClientID',            # Name of the new column storing the client IDs (e.g., 'MT_001')
        value_name='Consumption'        # Name of the new column storing the actual kW values
    )

    # Memory optimization: Downcast the target variable to save RAM
    df_long['Consumption'] = df_long['Consumption'].astype(np.float32)
    # Memory optimization: Convert ClientID to a category type 
    df_long['ClientID'] = df_long['ClientID'].astype('category')
        
    # 4. Cleaning
    df_long = clean_clients(df_long)
    
    # 5. Weather Merge
    print("Fetching and merging weather...")
    weather_df = get_national_weather()
    # Floor to hour to merge with weather
    df_long['Date_Hour'] = df_long['Date'].dt.floor('h')
    df_long = df_long.merge(weather_df, left_on='Date_Hour', right_on='Date', how='left', suffixes=('', '_w'))
    df_long = df_long.drop(columns=['Date_Hour', 'Date_w'])
    
    # 6. Lags & Rolling
    df_long = add_lags_and_rolling(df_long)
    
    # 7. Clustering (Shape and Volume)
    print("Applying Shape & Volume clustering mappings...")
    
    df_train = df_long[df_long['Date'].dt.year < 2014].copy()
    df_test  = df_long[df_long['Date'].dt.year >= 2014].copy()

    df_train, df_test = apply_profile_clustering(df_train, df_test, n_clusters=5)
    df_train, df_test = apply_volume_clustering(df_train, df_test)

    # Recombine train and test safely
    df_long = pd.concat([df_train, df_test]).sort_values(by=['ClientID', 'Date']).reset_index(drop=True)

    # 8. Export
    print(f"Exporting to {output_path}...")
    df_long.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    input_p = os.path.join(PROJECT_ROOT, "Datasets", "Electricity Dataset.csv")
    output_p = os.path.join(PROJECT_ROOT, "Datasets", "processed_electricity_data.parquet")
    process_data(input_p, output_p)