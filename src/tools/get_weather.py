import pandas as pd
import numpy as np
import requests

def get_national_weather(start_date="2011-01-01", end_date="2015-01-01"):
    """
    Fetches historical weather data from Open-Meteo and calculates population-weighted 
    national metrics (HDH and CDH) for Portugal.

    This function retrieves hourly 2m temperature data for the four major population 
    centers in Portugal. It computes a national average temperature weighted by the 
    2011 census population. From this average, it derives Heating Degree Hours (HDH) 
    and Cooling Degree Hours (CDH) using a base temperature of 18°C, which is the 
    standard threshold for energy demand and HVAC modeling.

    Args:
        start_date (str): Start date for the weather archive (YYYY-MM-DD). 
                          Defaults to "2011-01-01".
        end_date (str): End date for the weather archive (YYYY-MM-DD). 
                        Defaults to "2015-01-01".

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'Date': Datetime index/column.
            - 'HDH': Heating Degree Hours (calculated as max(0, 18 - Temp)).
            - 'CDH': Cooling Degree Hours (calculated as max(0, Temp - 18)).
    """
    
    # 2011 population estimates (thousands) for weighting
    # Source: INE Portugal
    cities = {
        "Lisbon": {"lat": 38.72, "lon": -9.14,  "population": 547},
        "Porto":  {"lat": 41.15, "lon": -8.61,  "population": 237},
        "Faro":   {"lat": 37.02, "lon": -7.93,  "population": 64},
        "Evora":  {"lat": 38.57, "lon": -7.91,  "population": 56},
    }

    total_population = sum(c["population"] for c in cities.values())

    all_temps = []

    # Base temperature for energy demand modeling (standard threshold)
    BASE_TEMP = 18.0

    for city, info in cities.items():
        print(f"  Fetching weather for {city}...")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   info["lat"],
            "longitude":  info["lon"],
            "start_date": start_date,
            "end_date":   end_date,
            "hourly":     "temperature_2m",
            "timezone":   "Europe/Lisbon",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Ensure the API call was successful
        data = response.json()

        weight = info["population"] / total_population

        temp_df = pd.DataFrame({
            "Date":   pd.to_datetime(data["hourly"]["time"]),
            "temp":   data["hourly"]["temperature_2m"],
        })
        temp_df["weighted_temp"] = temp_df["temp"] * weight
        all_temps.append(temp_df[["Date", "weighted_temp"]])

    # Aggregate all weighted temperatures by summing the lists of DataFrames
    national_temp = sum(df["weighted_temp"] for df in all_temps)
    
    weather_df = pd.DataFrame({
        "Date": all_temps[0]["Date"],
        "Temp_National_Avg": national_temp.astype(np.float32)
    })

    # Calculate Degree Hours using the defined base temperature
    weather_df["HDH"] = (BASE_TEMP - weather_df["Temp_National_Avg"]).clip(lower=0)
    weather_df["CDH"] = (weather_df["Temp_National_Avg"] - BASE_TEMP).clip(lower=0)

    return weather_df[["Date", "HDH", "CDH"]]