import pandas as pd
import numpy as np
from src.tools import add_temporal_features, get_national_weather, clean_clients


def process_data(input_path, output_path):
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(input_path, sep=';', decimal=',')

    # Rename timestamp column to 'Date' to keep naming consistent
    df = df.rename(columns={'Unnamed: 0': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)


    df = df.rename(columns={'Unnamed: 0': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
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

    
    # 4. Weather Merge
    print("Fetching and merging weather...")
    weather_df = get_national_weather()
    # Floor to hour to merge with weather
    df_long['Date_Hour'] = df_long['Date'].dt.floor('h')
    df_long = df_long.merge(weather_df, left_on='Date_Hour', right_on='Date', how='left', suffixes=('', '_w'))
    df_long = df_long.drop(columns=['Date_Hour', 'Date_w'])
    
    # 5. Cleaning
    df_long = clean_clients(df_long)
    
    # 6. Lags & Rolling
    df_long = add_lags_and_rolling(df_long)
    
    # 7. Export
    print(f"Exporting to {output_path}...")
    df_long.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    process_data("../../Datasets/raw_data.csv", "../../Datasets/processed_data.parquet")