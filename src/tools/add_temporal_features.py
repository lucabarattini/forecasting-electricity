import pandas as pd
import numpy as np
import datetime
from dateutil import easter
import requests
from src.tools.get_holidays import get_holidays

def add_temporal_features(df):
    """
    Adds time-based feature columns to the DataFrame based on the 'Date' column.
    """
    # Extract unique years from the dataset to calculate holidays dynamically
    unique_years = df['Date'].dt.year.unique()
    holidays_set = get_holidays(unique_years)

    new_features = {

        # 1=Monday to 7=Sunday
        'Weekday': df['Date'].dt.dayofweek + 1,

        # 1 to 24
        'Hour': df['Date'].dt.hour + 1,

        # 1 to 12
        'Month' : df['Date'].dt.month + 1,

        # True if Saturday (5) or Sunday (6)
        'Is_Weekend': df['Date'].dt.dayofweek >= 5,

        # True if the date is in the calculated holidays set
        # Evaluates national holidays dynamically. Essential for capturing anomalous consumption drops during public holidays
        'Is_Holiday': df['Date'].dt.date.isin(holidays_set)
    }
    
    
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    return df