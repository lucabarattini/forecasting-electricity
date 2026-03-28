"""
models/predict.py
-----------------
Inference functions for Linear Regression, Prophet, and SARIMAX.
"""

from __future__ import annotations
import os
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ARTIFACTS_DIR     = "./artifacts"
FORECAST_START_DT = pd.Timestamp("2015-01-01 00:00:00")
FREQ              = "15min"


@dataclass
class ForecastResult:
    model_name: str
    client_id: str
    horizon_hours: int
    timestamps: list[str]
    predictions_kw: list[float]
    error: str | None = None

    @property
    def n_steps(self) -> int:
        return len(self.predictions_kw)

    @property
    def total_kwh(self) -> float:
        return round(sum(v * 0.25 for v in self.predictions_kw), 3)

    @property
    def mean_kw(self) -> float:
        return round(float(np.mean(self.predictions_kw)), 3)

    @property
    def peak_kw(self) -> float:
        return round(float(np.max(self.predictions_kw)), 3)

    @property
    def peak_timestamp(self) -> str:
        idx = int(np.argmax(self.predictions_kw))
        return self.timestamps[idx]

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "client_id": self.client_id,
            "horizon_hours": self.horizon_hours,
            "steps": self.n_steps,
            "total_kwh": self.total_kwh,
            "mean_kw": self.mean_kw,
            "peak_kw": self.peak_kw,
            "peak_at": self.peak_timestamp,
            "forecast": [
                {"timestamp": ts, "kw": kw}
                for ts, kw in zip(self.timestamps, self.predictions_kw)
            ],
            "error": self.error,
        }

    def to_summary(self) -> str:
        if self.error:
            return f"[{self.model_name}] ERROR for {self.client_id}: {self.error}"
        return (
            f"--- {self.model_name} | {self.client_id} ---\n"
            f"Period  : {self.timestamps[0]}  to  {self.timestamps[-1]}\n"
            f"Total   : {self.total_kwh} kWh\n"
            f"Average : {self.mean_kw} kW\n"
            f"Peak    : {self.peak_kw} kW at {self.peak_timestamp}\n"
        )


def _load_artifact(client_id: str, model_name: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_{client_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No artifact for model='{model_name}' client='{client_id}'. "
            f"Run train_and_save.py first."
        )
    return joblib.load(path)


def _make_future_timestamps(horizon_hours: int) -> pd.DatetimeIndex:
    n_steps = horizon_hours * 4
    return pd.date_range(start=FORECAST_START_DT, periods=n_steps, freq=FREQ)


def _get_future_features(client_id: str, horizon_hours: int,
                          df_all: pd.DataFrame) -> pd.DataFrame:
    future_ts = _make_future_timestamps(horizon_hours)
    n_steps   = len(future_ts)

    df_c = df_all[df_all["ClientID"] == client_id].copy()
    df_c = df_c.set_index("Date").sort_index()

    one_year_ago_start = FORECAST_START_DT - pd.DateOffset(years=1)
    one_year_ago_end   = one_year_ago_start + pd.Timedelta(hours=horizon_hours)
    mirror = df_c.loc[one_year_ago_start:one_year_ago_end].copy()

    while len(mirror) < n_steps:
        mirror = pd.concat([mirror, mirror])
    mirror = mirror.iloc[:n_steps].copy()
    mirror.index = future_ts
    return mirror


def _get_recent_history(client_id: str, n: int, df_all: pd.DataFrame) -> list[float]:
    df_c = df_all[df_all["ClientID"] == client_id].sort_values("Date")
    return df_c["Consumption"].values[-n:].tolist()


def predict_linear_regression(client_id: str, horizon_hours: int,
                               df_all: pd.DataFrame) -> ForecastResult:
    future_ts = _make_future_timestamps(horizon_hours)
    ts_strs   = [str(ts) for ts in future_ts]
    n_steps   = len(future_ts)

    try:
        path = os.path.join(ARTIFACTS_DIR, "lr_cluster_models.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError("Cluster LR artifact not found. Please run the Linear Regression pipeline first.")
        
        art = joblib.load(path)
        client_clusters = art["client_clusters"]
        
        if client_id not in client_clusters:
            raise KeyError(f"Client {client_id} is not mapped to any cluster in the training data.")
            
        cluster_id     = client_clusters[client_id]
        model          = art["cluster_models"][cluster_id]
        scaler_target  = art["client_scalers"][client_id]
        scaler_weather = art["scaler_weather"]
        feature_cols   = art["feature_cols"]

        future_feat = _get_future_features(client_id, horizon_hours, df_all)
        future_feat["Is_Weekend"] = future_feat["Is_Weekend"].astype(int)
        future_feat["Is_Holiday"] = future_feat["Is_Holiday"].astype(int)

        future_model = pd.get_dummies(
            future_feat.reset_index(),
            columns=["Hour", "Weekday", "Consumer_Category"],
            drop_first=True
        )

        weather_cols = ["Temp_National_Avg", "HDH", "CDH"]
        future_model[weather_cols] = scaler_weather.transform(
            future_model[weather_cols].values
        )

        for col in feature_cols:
            if col not in future_model.columns:
                future_model[col] = 0
        future_model = future_model[feature_cols].fillna(0)

        history_buffer = _get_recent_history(client_id, 96, df_all)
        history_scaled = scaler_target.transform(
            np.array(history_buffer).reshape(-1, 1)
        ).flatten().tolist()

        idx_lag15 = feature_cols.index("Lag_15min_Scaled") if "Lag_15min_Scaled" in feature_cols else None
        idx_lag24 = feature_cols.index("Lag_24h_Scaled")   if "Lag_24h_Scaled"   in feature_cols else None
        idx_roll  = feature_cols.index("Rolling_Mean_4h_Scaled") if "Rolling_Mean_4h_Scaled" in feature_cols else None

        X = future_model.values.copy().astype(float)
        preds_scaled = []

        for i in range(n_steps):
            lag15 = history_scaled[-1]
            lag24 = history_scaled[-96] if len(history_scaled) >= 96 else lag15
            roll  = float(np.mean(history_scaled[-16:])) if len(history_scaled) >= 16 else lag15

            if idx_lag15 is not None: X[i, idx_lag15] = lag15
            if idx_lag24 is not None: X[i, idx_lag24] = lag24
            if idx_roll  is not None: X[i, idx_roll]  = roll

            pred = model.predict(X[[i]])[0]
            preds_scaled.append(pred)
            history_scaled.append(pred)

        preds_kw = scaler_target.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        preds_kw = [round(max(0.0, float(v)), 3) for v in preds_kw]

        return ForecastResult("Linear Regression", client_id, horizon_hours, ts_strs, preds_kw)

    except Exception as e:
        return ForecastResult("Linear Regression", client_id, horizon_hours, ts_strs, [], error=str(e))


def predict_prophet(client_id: str, horizon_hours: int,
                    df_all: pd.DataFrame) -> ForecastResult:
    future_ts = _make_future_timestamps(horizon_hours)
    ts_strs   = [str(ts) for ts in future_ts]
    n_steps   = len(future_ts)

    try:
        art       = _load_artifact(client_id, "prophet")
        m         = art["model"]
        history_y = list(art["history_y"])

        future_feat = _get_future_features(client_id, horizon_hours, df_all)

        # Single batch call instead of 96 individual calls
        future_df = pd.DataFrame({
            "ds": future_ts,
            "Temp_National_Avg": future_feat["Temp_National_Avg"].values,
            "Lag_15min":        [history_y[-1]] * n_steps,
            "Lag_24h":          [history_y[-96] if len(history_y) >= 96 else history_y[-1]] * n_steps,
            "Rolling_Mean_4h":  [float(np.mean(history_y[-16:])) if len(history_y) >= 16 else history_y[-1]] * n_steps,
        })

        forecast = m.predict(future_df)
        preds_kw = [max(0.0, round(float(v), 3)) for v in forecast["yhat"].values]

        return ForecastResult("Prophet", client_id, horizon_hours, ts_strs, preds_kw)

    except Exception as e:
        return ForecastResult("Prophet", client_id, horizon_hours, ts_strs, [], error=str(e))


def predict_sarimax(client_id: str, horizon_hours: int,
                    df_all: pd.DataFrame) -> ForecastResult:
    future_ts = _make_future_timestamps(horizon_hours)
    ts_strs   = [str(ts) for ts in future_ts]
    n_steps   = len(future_ts)

    try:
        art       = _load_artifact(client_id, "sarimax")
        result    = art["result"]
        exog_cols = art["exog_cols"]

        future_feat = _get_future_features(client_id, horizon_hours, df_all)
        exog_future = future_feat[exog_cols].astype(float).iloc[:n_steps]
        exog_future.index = future_ts

        y_pred   = result.forecast(steps=n_steps, exog=exog_future).values
        preds_kw = [round(max(0.0, float(v)), 3) for v in y_pred]

        return ForecastResult("SARIMAX", client_id, horizon_hours, ts_strs, preds_kw)

    except Exception as e:
        return ForecastResult("SARIMAX", client_id, horizon_hours, ts_strs, [], error=str(e))