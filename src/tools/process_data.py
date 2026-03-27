import pandas as pd
import numpy as np
from src.tools.add_temporal_features import add_temporal_features

def process_data(path):
    df = pd.read_csv(path, sep=';', decimal=',')

    # Rename timestamp column to 'Date' to keep naming consistent
    df = df.rename(columns={'Unnamed: 0': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    # Add temporal features BEFORE melting (Huge performance boost: calculates on 140k rows instead of 51M)
    print("Adding temporal features...")
    df = add_temporal_features(df)

    # Define the columns that should remain fixed (the time features)
    # All other columns (the 370 clients) will be melted into rows
    fixed_vars = ['Date', 'Weekday', 'Hour', 'Month', 'Is_Weekend', 'Is_Holiday']

    df_long = pd.melt(
    df, 
    id_vars=fixed_vars,             # The columns to keep fixed
    var_name='ClientID',            # Name of the new column storing the client IDs (e.g., 'MT_001')
    value_name='Consumption'        # Name of the new column storing the actual kW values
)

    # Memory optimization: Downcast the target variable to save RAM.
    df_long['Consumption'] = df_long['Consumption'].astype(np.float32)

    # Memory optimization: Convert ClientID to a category type (saves massive amounts of memory for repeated strings)
    df_long['ClientID'] = df_long['ClientID'].astype('category')

    return df_long