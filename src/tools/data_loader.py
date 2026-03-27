import pandas as pd

def load_raw_data(input_path):
    """
    Loads the raw electricity CSV, renames the date column, 
    converts to datetime, and sorts chronologically.
    """
    print(f"Loading data from: {input_path}...")
    df = pd.read_csv(input_path, sep=';', decimal=',')

    # Rename timestamp column to 'Date' to keep naming consistent
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df