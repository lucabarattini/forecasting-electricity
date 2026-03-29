"""
sarimax_model.py
----------------
Modularized implementation of the electricity load forecasting model using SARIMAX.
Trains one model per cluster (averaged shape) and scales/un-scales for individual clients.

Features:
- Dual-mode: 'long_term' (weather only) and 'day_ahead' (weather + lags).
- Cluster aggregation for performance.
- Safe merge logic for individual client mapping.
- SARIMAX(1, 1, 1)x(1, 1, 1, 24) configuration.
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import warnings
from src.tools.evaluation import compute_cluster_metrics

# Suppress statsmodels warnings for cleaner output
warnings.filterwarnings("ignore")

# Mapping environment for modular execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods

def load_processed_data(file_path):
    """Loads the processed data from a parquet file."""
    print("Loading processed data...")
    return pd.read_parquet(file_path)

def preprocess_and_split(df_long, mode='long_term'):
    """
    Groups data by Cluster/Date for training. 
    mode='long_term': Uses only calendar and weather.
    mode='day_ahead': Adds 24h and 1week consumption lags as exog.
    """
    print(f"Preparing train/test split and scaling (Mode: {mode.upper()})...")
    
    # 1. Train/Test Split (2014 threshold)
    train_raw = df_long[df_long['Date'].dt.year < 2014].copy()
    test_raw  = df_long[df_long['Date'].dt.year >= 2014].copy()

    # Ensure boolean flags are numerical for statsmodels
    train_raw['Is_Weekend'] = train_raw['Is_Weekend'].astype(np.float32)
    train_raw['Is_Holiday'] = train_raw['Is_Holiday'].astype(np.float32)
    test_raw['Is_Weekend'] = test_raw['Is_Weekend'].astype(np.float32)
    test_raw['Is_Holiday'] = test_raw['Is_Holiday'].astype(np.float32)
    
    # 2. Weather Scaling
    weather_cols = ['HDH', 'CDH', 'HDH_lag24h', 'CDH_lag24h', 'HDH_anomaly', 'CDH_anomaly']
    weather_cols = [c for c in weather_cols if c in df_long.columns]
    
    scaler_weather = StandardScaler()
    train_raw[weather_cols] = scaler_weather.fit_transform(train_raw[weather_cols])
    test_raw[weather_cols]  = scaler_weather.transform(test_raw[weather_cols])
    
    # Define exogenous regressors based on mode
    regressors = weather_cols + ['Is_Weekend', 'Is_Holiday']
    regressors = [c for c in regressors if c in df_long.columns]
    
    if mode == 'day_ahead':
        # Include consumption lags as exogenous variables
        # Note: We assume Lag_24h and Lag_1week exist in the dataset
        if 'Lag_24h' in df_long.columns and 'Lag_1week' in df_long.columns:
            regressors.extend(['Lag_24h_Scaled', 'Lag_1week_Scaled'])

    # 3. Per-Client Consumption (and Lag) Scaling
    client_scalers = {}
    print("Scaling individual clients (preventing leakage)...")
    
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

    # 4. Aggregation by Cluster for SARIMAX
    print("Aggregating data by Cluster for SARIMAX training...")
    agg_dict = {'Consumption_Scaled': 'mean'}
    for col in regressors:
        if 'Lag_' in col and 'Scaled' in col:
            agg_dict[col] = 'mean' # Lags are individual, we average them for the cluster
        else:
            agg_dict[col] = 'first' # Weather/Calendar is global, 'first' is faster
        
    train_agg = train_raw.groupby(['Cluster', 'Date'], observed=True).agg(agg_dict).reset_index()
    test_agg  = test_raw.groupby(['Cluster', 'Date'], observed=True).agg(agg_dict).reset_index()
    
    # Ensure time continuity for SARIMAX (mandatory)
    train_agg = train_agg.sort_values(['Cluster', 'Date'])
    test_agg  = test_agg.sort_values(['Cluster', 'Date'])
    
    return train_agg, test_agg, test_raw, client_scalers, scaler_weather, regressors
    

def train_models(train_agg, regressors):
    """
    Trains one SARIMAX model per cluster.
    SARIMAX is computationally heavy; we use simple orders to balance performance.
    """
    print(f"Training SARIMAX models for {train_agg['Cluster'].nunique()} clusters...")
    cluster_models = {}
    unique_clusters = sorted(train_agg['Cluster'].unique())

    for cluster_id in tqdm(unique_clusters, desc="Training"):
        df_cluster = train_agg[train_agg['Cluster'] == cluster_id].set_index('Date')

        df_cluster = df_cluster.resample('1h').mean().ffill()
        
        endog = df_cluster['Consumption_Scaled'].astype(np.float64)
        exog = df_cluster[regressors].astype(np.float64)
        
        model = SARIMAX(
            endog=endog,
            exog=exog,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 24), 
            enforce_stationarity=False,
            enforce_invertibility=False,
            low_memory=True
        )
        
        results = model.fit(disp=False)
        cluster_models[cluster_id] = results
        
    return cluster_models


def predict_models(cluster_models, test_agg, test_raw, client_scalers, regressors):
    """
    Generates cluster-level forecasts and maps them safely back to individual clients.
    """
    print("Generating forecasts and un-scaling to raw kW...")
    
    all_cluster_forecasts = []
    
    for cluster_id, results in cluster_models.items():
        df_test_c = test_agg[test_agg['Cluster'] == cluster_id].set_index('Date')
        
        if len(df_test_c) > 0:
            df_test_hourly = df_test_c.resample('1h').mean().ffill()

            exog_test = df_test_hourly[regressors].astype(np.float64)
            
            forecast = results.get_forecast(steps=len(df_test_hourly), exog=exog_test)
            
            fcst_hourly = pd.DataFrame({
                'Cluster': cluster_id,
                'Date': df_test_hourly.index,
                'Predicted_Consumption_Scaled': forecast.predicted_mean.values
            }).set_index('Date')
            
            fcst_15min = fcst_hourly.resample('15min').ffill().reset_index()
            
            all_cluster_forecasts.append(fcst_15min)

    # 2. Safe Merge to Individual Clients
    global_forecasts = pd.concat(all_cluster_forecasts, ignore_index=True)
    test_raw = test_raw.drop(columns=['Predicted_Consumption_Scaled'], errors='ignore')
    test_raw = test_raw.merge(global_forecasts, on=['Cluster', 'Date'], how='left')
    
    # 3. Inverse Scaling
    test_raw['Actual_kW'] = test_raw['Consumption'] # Spostato fuori (operazione vettoriale veloce)
    test_raw['Predicted_kW'] = np.nan
    
    for client in tqdm(test_raw['ClientID'].unique(), desc="Un-scaling Clients"):
        mask = test_raw['ClientID'] == client
        if mask.any() and client in client_scalers:
            scaler = client_scalers[client]
            preds_scaled = test_raw.loc[mask, 'Predicted_Consumption_Scaled'].values.reshape(-1, 1)
            
            # Filtro rapido per evitare di processare blocchi di soli NaN
            if pd.isna(preds_scaled).all():
                continue
                
            unscaled = scaler.inverse_transform(preds_scaled).flatten()
            test_raw.loc[mask, 'Predicted_kW'] = np.maximum(unscaled, 0)
                
    return test_raw


def evaluate_models(test_raw):
    """
    Computes business-oriented performance metrics (MAPE/WMAPE) at portfolio level.
    """
    print("\nEvaluating Portfolio Performance...")
    
    portfolio_eval = (
        test_raw.dropna(subset=['Actual_kW', 'Predicted_kW'])
        .groupby(['Cluster', 'Date'], observed=True)[['Actual_kW', 'Predicted_kW']]
        .sum()
        .reset_index()
    )

    summary = compute_cluster_metrics(portfolio_eval)

    return portfolio_eval, summary
    

def save_sarimax_artifacts(cluster_models, client_scalers, scaler_weather, regressors, mode, artifacts_dir=None):
    """Saves the models and scalers for future production inference."""
    if artifacts_dir is None:
        artifacts_dir = os.path.join(PROJECT_ROOT, 'agent', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    file_name = f"sarimax_cluster_{mode}.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    artifact = {
        "cluster_models": cluster_models,
        "client_scalers": client_scalers,
        "scaler_weather": scaler_weather,
        "regressors": regressors,
        "mode": mode
    }
    joblib.dump(artifact, path)
    print(f"\n SARIMAX ({mode}) artifacts successfully saved to: {path}")

def run_sarimax_pipeline(file_path, mode='long_term', plot=False):
    """Complete workflow for SARIMAX forecasting."""
    df_long = load_processed_data(file_path)
    
    train_agg, test_agg, test_raw, client_scalers, scaler_weather, regressors = preprocess_and_split(df_long, mode=mode)
    cluster_models = train_models(train_agg, regressors)
    
    test_raw = predict_models(cluster_models, test_agg, test_raw, client_scalers, regressors)
    portfolio_eval, summary = evaluate_models(test_raw)
    
    save_sarimax_artifacts(cluster_models, client_scalers, scaler_weather, regressors, mode)
    
    if plot:
        plot_cluster_portfolio(portfolio_eval, summary, model_label=f"SARIMAX ({mode.replace('_', ' ').title()})")
        analyze_time_periods(test_raw)
        
    return cluster_models, test_raw, portfolio_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "Datasets", "processed_electricity_data.parquet")
    
    print("=== RUNNING SARIMAX LONG TERM FORECAST ===")
    run_sarimax_pipeline(DATA_PATH, mode='long_term', plot=True)
