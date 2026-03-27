import pandas as pd
import numpy as np
import datetime
from dateutil import easter
import requests
from src.tools.get_holidays import get_holidays

def add_temporal_features(df):
    """
    Performs feature engineering by extracting time-based components from the 'Date' column.
    
    This function calculates calendar-based features such as hour of the day, day of the week, 
    and month. It also identifies weekends and national holidays, which are critical 
    for capturing cyclical consumption patterns and anomalous drops during non-working days.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain a 'Date' column in datetime format.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus the following columns:
            - 'Weekday': Day of the week (1=Monday, 7=Sunday).
            - 'Hour': Hour of the day (1 to 24).
            - 'Month': Month of the year (1 to 12).
            - 'Is_Weekend': Boolean flag (True for Saturday and Sunday).
            - 'Is_Holiday': Boolean flag (True if the date is a national holiday).
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