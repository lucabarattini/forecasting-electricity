"""
prophet_model.py
----------------
Modularized implementation of the electricity load forecasting model using Facebook Prophet.
Trains one model per cluster (averaged shape) and scales/un-scales for individual clients.

Features:
- Global weather scaling (fitted on train only to prevent leakage).
- Per-client consumption scaling.
- Portuguese holidays and external regressors.
- Performance evaluation at the portfolio level.
"""

import os
import sys
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import logging

# Mapping environment for modular execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods

# Suppress Prophet/cmdstanpy verbose logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

def load_processed_data(file_path):
    """Loads the processed data from a parquet file."""
    print("Loading processed data...")
    return pd.read_parquet(file_path)

def preprocess_and_split(df_long, mode='long_term'):
    """
    Groups data by Cluster/Date for training. 
    mode='long_term': Uses only calendar and weather.
    mode='day_ahead': Adds 24h and 1week consumption lags.
    """
    print(f"Preparing train/test split and scaling (Mode: {mode.upper()})...")
    
    train_raw = df_long[df_long['Date'].dt.year < 2014].copy()
    test_raw  = df_long[df_long['Date'].dt.year >= 2014].copy()
    
    # 1. Weather Scaling
    weather_cols = ['HDH', 'CDH', 'HDH_lag24h', 'CDH_lag24h', 'HDH_anomaly', 'CDH_anomaly']
    weather_cols = [c for c in weather_cols if c in df_long.columns]
    
    scaler_weather = StandardScaler()
    train_raw[weather_cols] = scaler_weather.fit_transform(train_raw[weather_cols])
    test_raw[weather_cols]  = scaler_weather.transform(test_raw[weather_cols])
    
    # Define regressors based on mode
    regressors = weather_cols.copy()
    if mode == 'day_ahead':
        regressors.extend(['Lag_24h_Scaled', 'Lag_1week_Scaled'])

    # 2. Per-Client Consumption (and Lag) Scaling
    client_scalers = {}
    print("Scaling individual clients...")
    
    # Initialize columns for speed
    cols_to_scale = ['Consumption_Scaled']
    if mode == 'day_ahead':
        cols_to_scale.extend(['Lag_24h_Scaled', 'Lag_1week_Scaled'])
    for col in cols_to_scale:
        train_raw[col] = np.nan
        test_raw[col] = np.nan

    for client in tqdm(df_long['ClientID'].unique(), desc="Scaling Clients"):
        scaler = StandardScaler()
        train_mask = train_raw['ClientID'] == client
        test_mask  = test_raw['ClientID'] == client
        
        if not train_mask.any():
            continue
            
        train_raw.loc[train_mask, 'Consumption_Scaled'] = scaler.fit_transform(
            train_raw.loc[train_mask, 'Consumption'].values.reshape(-1, 1)
        ).flatten()
        
        if mode == 'day_ahead':
            train_raw.loc[train_mask, 'Lag_24h_Scaled'] = scaler.transform(train_raw.loc[train_mask, 'Lag_24h'].values.reshape(-1, 1)).flatten()
            train_raw.loc[train_mask, 'Lag_1week_Scaled'] = scaler.transform(train_raw.loc[train_mask, 'Lag_1week'].values.reshape(-1, 1)).flatten()
        
        if test_mask.any():
            test_raw.loc[test_mask, 'Consumption_Scaled'] = scaler.transform(
                test_raw.loc[test_mask, 'Consumption'].values.reshape(-1, 1)
            ).flatten()
            
            if mode == 'day_ahead':
                test_raw.loc[test_mask, 'Lag_24h_Scaled'] = scaler.transform(test_raw.loc[test_mask, 'Lag_24h'].values.reshape(-1, 1)).flatten()
                test_raw.loc[test_mask, 'Lag_1week_Scaled'] = scaler.transform(test_raw.loc[test_mask, 'Lag_1week'].values.reshape(-1, 1)).flatten()
            
        client_scalers[client] = scaler

    # Drop NaNs created by lags if in day-ahead mode
    if mode == 'day_ahead':
        train_raw = train_raw.dropna(subset=['Lag_24h_Scaled', 'Lag_1week_Scaled'])

    # 3. Aggregation by Cluster for Prophet
    print("Aggregating data by Cluster for Prophet training...")
    agg_dict = {'Consumption_Scaled': 'mean'}
    for col in regressors:
        if 'Lag_' in col and 'Scaled' in col:
            agg_dict[col] = 'mean' # Lags are individual, so we average them for the cluster
        else:
            agg_dict[col] = 'first' # Weather is global, 'first' is faster
        
    train_agg = train_raw.groupby(['Cluster', 'Date'], observed=True).agg(agg_dict).reset_index()
    test_agg  = test_raw.groupby(['Cluster', 'Date'], observed=True).agg(agg_dict).reset_index()
    
    train_agg = train_agg.rename(columns={'Date': 'ds', 'Consumption_Scaled': 'y'})
    test_agg  = test_agg.rename(columns={'Date': 'ds', 'Consumption_Scaled': 'y'})
    
    return train_agg, test_agg, test_raw, client_scalers, scaler_weather, regressors
    

def train_models(train_agg, regressors):
    """
    Trains one Facebook Prophet model per cluster using Portuguese holidays 
    and external regressors (weather + optional lags).
    """
    print(f"Training Prophet models for {train_agg['Cluster'].nunique()} clusters...")
    cluster_models = {}
    unique_clusters = sorted(train_agg['Cluster'].unique())

    for cluster_id in tqdm(unique_clusters, desc="Training"):
        df_cluster = train_agg[train_agg['Cluster'] == cluster_id]
        
        m = Prophet(
            changepoint_prior_scale=0.05, 
            uncertainty_samples=0, 
            daily_seasonality=False
        )
        m.add_seasonality(name='daily', period=1, fourier_order=15) 
        m.add_country_holidays(country_name='PT')
        
        for reg in regressors:
            m.add_regressor(reg)
            
        m.fit(df_cluster)
        cluster_models[cluster_id] = m
        
    return cluster_models


def predict_models(cluster_models, test_agg, test_raw, client_scalers, regressors):
    """
    Generates cluster-level forecasts and maps them safely back to individual clients.
    """
    print("Generating forecasts and un-scaling to raw kW...")
    
    # 1. Cluster-level Forecast (Vectorized)
    all_cluster_forecasts = []
    for cluster_id, model in cluster_models.items():
        df_test_c = test_agg[test_agg['Cluster'] == cluster_id]
        if len(df_test_c) > 0:
            future = df_test_c[['ds'] + regressors].copy()
            forecast = model.predict(future)
            
            fcst_df = pd.DataFrame({
                'Cluster': cluster_id,
                'Date': forecast['ds'],
                'Predicted_Consumption_Scaled': forecast['yhat']
            })
            all_cluster_forecasts.append(fcst_df)

    global_forecasts = pd.concat(all_cluster_forecasts, ignore_index=True)

    # 2. Safe Merge to Individual Clients
    test_raw = test_raw.drop(columns=['Predicted_Consumption_Scaled'], errors='ignore')
    test_raw = test_raw.merge(global_forecasts, on=['Cluster', 'Date'], how='left')
    
    # 3. Inverse Scaling
    test_raw['Predicted_kW'] = np.nan
    for client in tqdm(test_raw['ClientID'].unique(), desc="Un-scaling Clients"):
        mask = test_raw['ClientID'] == client
        if mask.any() and client in client_scalers:
            scaler = client_scalers[client]
            unscaled = scaler.inverse_transform(
                test_raw.loc[mask, 'Predicted_Consumption_Scaled'].values.reshape(-1, 1)
            ).flatten()
            
            test_raw.loc[mask, 'Predicted_kW'] = np.maximum(unscaled, 0)
            test_raw.loc[mask, 'Actual_kW'] = test_raw.loc[mask, 'Consumption']
                
    return test_raw


def evaluate_models(test_raw):
    """
    Computes business-oriented performance metrics (MAPE/WMAPE) at portfolio level.
    Delegates all mathematical logic to the centralized evaluation module.
    """
    print("\nEvaluating Portfolio Performance...")
    
    # Portfolio aggregation by Cluster and Date
    portfolio_eval = (
        test_raw.dropna(subset=['Actual_kW', 'Predicted_kW'])
        .groupby(['Cluster', 'Date'], observed=True)[['Actual_kW', 'Predicted_kW']]
        .sum()
        .reset_index()
    )
    
    # Metrics Calculation via centralized tools
    summary = compute_cluster_metrics(portfolio_eval)

    return portfolio_eval, summary

def save_prophet_artifacts(cluster_models, client_scalers, scaler_weather, regressors, mode, artifacts_dir=None):
    if artifacts_dir is None:
        artifacts_dir = os.path.join(PROJECT_ROOT, 'agent', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    file_name = f"prophet_cluster_{mode}.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    artifact = {
        "cluster_models": cluster_models,
        "client_scalers": client_scalers,
        "scaler_weather": scaler_weather,
        "regressors": regressors,
        "mode": mode
    }
    joblib.dump(artifact, path)
    print(f"\n Prophet ({mode}) artifacts successfully saved to: {path}")

def run_prophet_pipeline(file_path, mode='long_term', plot=False):
    """
    Complete workflow. Mode can be 'long_term' or 'day_ahead'.
    """
    df_long = load_processed_data(file_path)
    
    train_agg, test_agg, test_raw, client_scalers, scaler_weather, regressors = preprocess_and_split(df_long, mode=mode)
    cluster_models = train_models(train_agg, regressors)
    
    test_raw = predict_models(cluster_models, test_agg, test_raw, client_scalers, regressors)
    portfolio_eval, summary = evaluate_models(test_raw)
    
    save_prophet_artifacts(cluster_models, client_scalers, scaler_weather, regressors, mode)
    
    if plot:
        plot_cluster_portfolio(portfolio_eval, summary, model_label=f"Prophet ({mode.replace('_', ' ').title()})")
        analyze_time_periods(test_raw)
        
    return cluster_models, test_raw, portfolio_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "Datasets", "processed_electricity_data.parquet")
    
    run_prophet_pipeline(DATA_PATH, mode='long_term', plot=False)
    run_prophet_pipeline(DATA_PATH, mode='day_ahead', plot=False)