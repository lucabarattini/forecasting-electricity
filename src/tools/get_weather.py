import pandas as pd
import numpy as np
import requests

def get_national_weather(start_date="2011-01-01", end_date="2015-01-01"):
    
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
        data = response.json()

        weight = info["population"] / total_population

        temp_df = pd.DataFrame({
            "Date":   pd.to_datetime(data["hourly"]["time"]),
            "temp":   data["hourly"]["temperature_2m"],
        })
        temp_df["weighted_temp"] = temp_df["temp"] * weight
        all_temps.append(temp_df[["Date", "weighted_temp"]])

    # Sum weighted temperatures across all cities
    weather_df = all_temps[0].copy()
    weather_df["Temp_National_Avg"] = sum(df["weighted_temp"] for df in all_temps)
    weather_df = weather_df[["Date", "Temp_National_Avg"]]

    # Heating/Cooling Degree Hours (base 18°C is standard for energy modeling)
    weather_df["HDH"] = (18 - weather_df["Temp_National_Avg"]).clip(lower=0).astype(np.float32)
    weather_df["CDH"] = (weather_df["Temp_National_Avg"] - 18).clip(lower=0).astype(np.float32)
    weather_df["Temp_National_Avg"] = weather_df["Temp_National_Avg"].astype(np.float32)

    return weather_df[["Date", "HDH", "CDH"]]