"""
agent/inference/predict.py
--------------------------
Robust inference module for LR, Prophet, SARIMAX, and Non-Stationary Transformer (NST).
Supports absolute pathing and automatic feature alignment (dummification + scaling).
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Torch is required for NST
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

warnings.filterwarnings("ignore")

# Absolute path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
ARTIFACTS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "artifacts")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.models.transformer_model import NonStationaryTransformer
except ImportError:
    NonStationaryTransformer = None

FORECAST_START_DT = pd.Timestamp("2015-01-01 00:00:00")
FREQ              = "15min"

@dataclass
class ForecastResult:
    model_name: str
    client_id: str
    mode: str
    timestamps: List[str]
    predictions_kw: List[float]
    error: Optional[str] = None

    @property
    def total_kwh(self) -> float:
        return round(sum(v * 0.25 for v in self.predictions_kw), 3) if self.predictions_kw else 0.0

    @property
    def mean_kw(self) -> float:
        return round(float(np.mean(self.predictions_kw)), 3) if self.predictions_kw else 0.0

    @property
    def peak_kw(self) -> float:
        return round(float(np.max(self.predictions_kw)), 3) if self.predictions_kw else 0.0

    @property
    def peak_timestamp(self) -> str:
        if not self.predictions_kw: return "N/A"
        idx = int(np.argmax(self.predictions_kw))
        return self.timestamps[idx]

    def to_summary(self) -> str:
        if self.error:
            return f"[{self.model_name}] ERROR for {self.client_id}: {self.error}"
        return (
            f"--- {self.model_name.upper()} | {self.client_id} | {self.mode.upper()} ---\n"
            f"Period  : {self.timestamps[0]}  to  {self.timestamps[-1]}\n"
            f"Total   : {self.total_kwh} kWh\n"
            f"Average : {self.mean_kw} kW\n"
            f"Peak    : {self.peak_kw} kW at {self.peak_timestamp}\n"
        )

# ── HELPERS ──────────────────────────────────────────────────────────────────

def _load_cluster_artifact(model: str, mode: str) -> dict:
    """Loads a cluster-based artifact from agent/artifacts/ with absolute pathing."""
    filename = f"{model}_cluster_{mode}.pkl"
    path = os.path.join(ARTIFACTS_DIR, filename)
    
    # Fallback search for older non-mode-specific naming
    if not os.path.exists(path):
        global_filename = f"{model}_cluster_models.pkl"
        global_path = os.path.join(ARTIFACTS_DIR, global_filename)
        if os.path.exists(global_path):
            path = global_path
        else:
            raise FileNotFoundError(f"Artifact not found at {path} or {global_path}")
    
    return joblib.load(path)

def _get_future_features(client_id: str, horizon_hours: int, df_all: pd.DataFrame) -> pd.DataFrame:
    """Generates features by mirroring data from exactly one year ago and aligning to future dates."""
    n_steps   = horizon_hours * 4
    future_ts = pd.date_range(start=FORECAST_START_DT, periods=n_steps, freq=FREQ)

    df_c = df_all[df_all["ClientID"] == client_id].copy()
    if df_c.empty:
        raise ValueError(f"ClientID {client_id} not found in dataset.")
        
    df_c = df_c.set_index("Date").sort_index()

    one_year_ago_start = FORECAST_START_DT - pd.DateOffset(years=1)
    one_year_ago_end   = one_year_ago_start + pd.Timedelta(hours=horizon_hours)
    
    mirror = df_c.loc[one_year_ago_start:one_year_ago_end].copy()

    while len(mirror) < n_steps:
        mirror = pd.concat([mirror, mirror])
    
    mirror = mirror.iloc[:n_steps].copy()
    mirror.index = future_ts
    
    if "Hour" in mirror.columns:
        mirror['Hour'] = mirror.index.hour
    if "Weekday" in mirror.columns:
        mirror['Weekday'] = mirror.index.weekday
    if "Month" in mirror.columns:
        mirror['Month'] = mirror.index.month
    if "Is_Weekend" in mirror.columns:
        mirror['Is_Weekend'] = (mirror['Weekday'] >= 5).astype(float)
     
    return mirror


def _align_features(df: pd.DataFrame, expected_cols: List[str], target_scaler: Any = None) -> pd.DataFrame:
    """Performs dummification and scaling to align with model expectations."""
    # 1. Expand standard Categoricals
    df_aligned = pd.get_dummies(
        df.reset_index(),
        columns=["Hour", "Weekday", "Consumer_Category", "Month"],
        drop_first=True
    )
    
    # 2. Scale Lags if requested
    if target_scaler is not None:
        lag_cols = [c for c in df_aligned.columns if "Lag_" in c and "_Scaled" not in c]
        for c in lag_cols:
            scaled_name = f"{c}_Scaled"
            if scaled_name in expected_cols:
                df_aligned[scaled_name] = target_scaler.transform(df_aligned[[c]].values)
        
        # Also handle Rolling Mean
        if "Rolling_Mean_4h_Scaled" in expected_cols and "Rolling_Mean_4h" in df_aligned.columns:
            df_aligned["Rolling_Mean_4h_Scaled"] = target_scaler.transform(df_aligned[["Rolling_Mean_4h"]].values)

    # 3. Reindex to match expected columns precisely, filling missing with 0
    df_aligned = df_aligned.reindex(columns=expected_cols, fill_value=0)
    return df_aligned

# ── INFERENCE ──────────────────────────────────────────────────────────────────

def predict_power(client_id: str, model_name: str, mode: str, df_all: pd.DataFrame, horizon_hours: int = 48) -> ForecastResult:
    """Unified inference engine with robust feature alignment."""
    model_name = model_name.lower()
    mode = mode.lower()
    future_ts = pd.date_range(start=FORECAST_START_DT, periods=horizon_hours*4, freq=FREQ)
    ts_strs = [str(ts) for ts in future_ts]

    try:
        # 1. Fetch Cluster ID
        client_data = df_all[df_all["ClientID"] == client_id]
        if client_data.empty:
            raise KeyError(f"ClientID '{client_id}' not found.")
        cluster_id = client_data["Cluster"].iloc[0]

        # 2. Load Artifact
        art = _load_cluster_artifact(model_name, mode)
        client_scalers = art["client_scalers"]
        scaler_weather = art["scaler_weather"]
        
        # Identify feature list (varies by model type)
        expected_features = art.get("feature_cols") or art.get("regressors") or art.get("exog_cols") or []
        
        if client_id not in client_scalers:
            raise KeyError(f"Client '{client_id}' has no scaler.")
        
        scaler_target = client_scalers[client_id]

        # 3. Build Features
        future_feat = _get_future_features(client_id, horizon_hours, df_all)
        
        # Weather Scaling
        weather_cols = ["HDH", "CDH", "Temp_National_Avg", "HDH_lag24h", "CDH_lag24h", "HDH_anomaly", "CDH_anomaly"]
        weather_to_scale = [c for c in weather_cols if c in future_feat.columns]
        if weather_to_scale:
            try:
                # Some scalers might only support a subset, we handle gracefully
                future_feat[weather_to_scale] = scaler_weather.transform(future_feat[weather_to_scale])
            except:
                pass

        # Alignment (Dummies + Lags)
        X_df = _align_features(future_feat, expected_features, scaler_target)

        # 4. Model Inference
        if model_name == 'nst':
            if torch is None: raise ImportError("PyTorch not found.")
            if NonStationaryTransformer is None: raise ImportError("NST class not found.")
            
            # Instantiate model with the exact training parameters
            model_obj = NonStationaryTransformer(
                enc_in=len(expected_features), c_out=1,
                seq_len=168, label_len=24, pred_len=24
            )
            model_obj.load_state_dict(art["cluster_states"][cluster_id])
            model_obj.eval()

            # --- REAL PYTORCH INFERENCE ---
            # 1. Get the last 168 hours of real historical data for this client
            df_c = df_all[df_all["ClientID"] == client_id].copy()
            df_c = df_c.set_index("Date").sort_index()
            
            # Pre-scale Consumption and Weather to match training state
            df_c["Consumption"] = scaler_target.transform(df_c[["Consumption"]].values)
            weather_to_scale = [c for c in ["HDH", "CDH", "Temp_National_Avg", "HDH_lag24h", "CDH_lag24h", "HDH_anomaly", "CDH_anomaly"] if c in df_c.columns]
            if weather_to_scale:
                try: df_c[weather_to_scale] = scaler_weather.transform(df_c[weather_to_scale])
                except: pass

            # STRICTLY select only the expected features to avoid pandas Category aggregation errors
            cols_to_resample = [c for c in expected_features if c in df_c.columns]
            df_hourly = df_c[cols_to_resample].resample('1h').mean(numeric_only=True).ffill().tail(168)
            
            # Fill any missing expected features with 0
            for c in expected_features:
                if c not in df_hourly.columns:
                    df_hourly[c] = 0.0
                    
            X_hist = df_hourly[expected_features]
            
            # Create Time Features for the Transformer (-0.5 to 0.5 range)
            def _time_features(dates):
                h = dates.hour.values / 23.0 - 0.5
                d = dates.dayofweek.values / 6.0 - 0.5
                return np.stack([h, d], axis=-1).astype(np.float32)

            # Build Encoder Tensors
            enc_x = torch.tensor(X_hist.values, dtype=torch.float32).unsqueeze(0)
            enc_mark = torch.tensor(_time_features(df_hourly.index), dtype=torch.float32).unsqueeze(0)
            
            # Build Decoder Tensors (last 24h history + 24h zeros for predictions)
            dec_x_hist = X_hist.values[-24:]
            dec_x_zeros = np.zeros((24, len(expected_features)))
            dec_x = torch.tensor(np.vstack([dec_x_hist, dec_x_zeros]), dtype=torch.float32).unsqueeze(0)
            
            future_dates_1h = pd.date_range(start=FORECAST_START_DT, periods=24, freq='1h')
            dec_dates = df_hourly.index[-24:].append(future_dates_1h)
            dec_mark = torch.tensor(_time_features(dec_dates), dtype=torch.float32).unsqueeze(0)

            # Execute Forward Pass
            with torch.no_grad():
                output = model_obj(enc_x, enc_mark, dec_x, dec_mark)
                preds_scaled_1h = output.squeeze(0).cpu().numpy()[:, 0]
            
            # Upsample 24h output to 15-min intervals (96 steps) and tile for long horizons
            preds_scaled = np.repeat(preds_scaled_1h, 4)
            if len(preds_scaled) < len(future_ts):
                tiles = int(np.ceil(len(future_ts) / len(preds_scaled)))
                preds_scaled = np.tile(preds_scaled, tiles)[:len(future_ts)]
            else:
                preds_scaled = preds_scaled[:len(future_ts)]
            # ------------------------------
            
        elif model_name in ['lr', 'prophet', 'sarimax']:
            model_obj = art["cluster_models"][cluster_id]
            
            # 1. Safely remove target variables from exogenous features
            exog_cols = [c for c in expected_features if c not in ['Consumption', 'Consumption_Scaled']]
            
            # 2. Force to float to avoid Category/Boolean dtype errors in statsmodels
            exog_df = X_df[exog_cols].astype(float)
            
            if model_name == 'lr':
                # Use .values to strip pandas index completely
                preds_scaled = model_obj.predict(exog_df.values)
                
            elif model_name == 'prophet':
                future_df = exog_df.copy()
                future_df['ds'] = future_ts
                forecast = model_obj.predict(future_df)
                preds_scaled = forecast['yhat'].values
                
            elif model_name == 'sarimax':
                exog_hourly = exog_df.groupby(np.arange(len(exog_df)) // 4).mean().values
                preds_scaled_1h = model_obj.forecast(steps=horizon_hours, exog=exog_hourly).values
                preds_scaled = np.repeat(preds_scaled_1h, 4)[:len(future_ts)]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 5. Inverse Scale
        preds_kw = scaler_target.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        preds_kw = [round(max(0.0, float(v)), 3) for v in preds_kw]

        return ForecastResult(model_name, client_id, mode, ts_strs, preds_kw)

    except Exception as e:
        return ForecastResult(model_name, client_id, mode, ts_strs, [], error=str(e))