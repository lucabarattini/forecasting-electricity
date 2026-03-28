import pandas as pd

def load_raw_data(input_path):
    """
    Loads raw electricity data from a CSV file, standardizes the timestamp column, 
    and ensures chronological ordering.

    Args:
        input_path (str): Path to the source CSV file.

    Returns:
        pd.DataFrame: A cleaned DataFrame with a 'Date' column in datetime format, 
                      sorted from oldest to newest.
    """
    print(f"Loading data from: {input_path}...")
    df = pd.read_csv(input_path, sep=';', decimal=',')

    if 'Date' not in df.columns and 'Unnamed: 0' not in df.columns:
        raise KeyError(f"Critical error: No timestamp column found in {input_path}")

    # Rename timestamp column to 'Date' to keep naming consistent
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df = df.drop_duplicates(subset='Date').reset_index(drop=True)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df