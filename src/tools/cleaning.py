import pandas as pd
import numpy as np

def clean_clients(df):
    # This cell should take around 10 seconds to run

    # PART 1 REMOVE TRIMMING LEADING ZEROS
    print("Trimming leading zeros (finding actual start date for each client)...")

    # Identify the first timestamp with consumption > 0 for each client
    start_dates = df[df['Consumption'] > 0].groupby('ClientID', observed=True)['Date'].min().reset_index()
    start_dates.columns = ['ClientID', 'StartDate']

    # Merge the start dates back into the main dataframe
    df = df.merge(start_dates, on='ClientID')

    # Keep only the rows where Date is greater than or equal to the StartDate
    df = df[df['Date'] >= df['StartDate']].copy()
    df = df.drop(columns=['StartDate'])
    print(f"Trimmed inactive periods. Remaining rows: {len(df)}")

    # PART 2 REMOVE INACTIVE CLIENTS
    print("Removing inactive clients (near 0 consumption in the last 90 days)...")

    # Slelect the last month of data and calculate the total consumption per client
    max_date = df['Date'].max()
    cutoff_date = max_date - pd.Timedelta(days=30)
    last_month_df = df[df['Date'] >= cutoff_date]
    recent_consumption = last_month_df.groupby('ClientID')['Consumption'].sum()

    # Identify active clients
    active_clients = recent_consumption[recent_consumption > 1].index
    inactive_count = len(recent_consumption) - len(active_clients)
    print(f"Detected {inactive_count} inactive clients in the last month.")

    # Keep only historically active clients
    df = df[df['ClientID'].isin(active_clients)].copy()
    df['ClientID'] = df['ClientID'].cat.remove_unused_categories()
    print(f"Removed inactive clients. Remaining rows: {len(df)}")

    return df
