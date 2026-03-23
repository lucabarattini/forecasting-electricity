'''
Reads the parquet dataset, trains all three models per client, and saves them as .pkl files in artifacts/.
'''
from __future__ import annotations
import argparse, os, random, warnings, logging
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

PARQUET_PATH        = r"..\Datasets\processed_electricity_data.parquet"
ARTIFACTS_DIR       = r".\artifacts"
FORECAST_HORIZON    = 96
SARIMAX_TRAIN_WEEKS = 2
SARIMAX_ORDER       = (1, 0, 1)
SARIMAX_SEASONAL    = (1, 0, 1, 96)
EXOG_COLS_SARIMAX   = ["Weekday","Hour","Is_Weekend","Is_Holiday","Temp_National_Avg","HDH","CDH"]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data(parquet_path=PARQUET_PATH):
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df["Is_Weekend"] = df["Is_Weekend"].astype(int)
    df["Is_Holiday"] = df["Is_Holiday"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["ClientID","Date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows | {df['ClientID'].nunique()} unique clients")
    return df

# Train a linear regression model for a given client. 
def train_linear_regression(client_id, df_all):
    df_c = df_all[df_all["ClientID"]==client_id].copy().sort_values("Date")
    df_c = df_c.dropna(subset=["Lag_15min","Lag_24h","Rolling_Mean_4h"])

    df_model = pd.get_dummies(df_c, columns=["Hour","Weekday","Consumer_Category"], drop_first=True)

    cutoff = df_model["Date"].max() - pd.Timedelta(hours=24)
    train  = df_model[df_model["Date"] < cutoff].copy()

    weather_cols   = ["Temp_National_Avg","HDH","CDH"]
    scaler_weather = StandardScaler()
    train[weather_cols] = scaler_weather.fit_transform(train[weather_cols])

    scaler_target = StandardScaler()
    train["Consumption_Scaled"]      = scaler_target.fit_transform(train[["Consumption"]]).flatten()
    train["Lag_15min_Scaled"]        = scaler_target.transform(train[["Lag_15min"]].values.reshape(-1,1)).flatten()
    train["Lag_24h_Scaled"]          = scaler_target.transform(train[["Lag_24h"]].values.reshape(-1,1)).flatten()
    train["Rolling_Mean_4h_Scaled"]  = scaler_target.transform(train[["Rolling_Mean_4h"]].values.reshape(-1,1)).flatten()
    exclude = {"ClientID","Date","DayMonth","Consumption","Consumption_Scaled",
            "Lag_15min","Lag_24h","Rolling_Mean_4h"}
    feature_cols = [c for c in train.columns if c not in exclude]

    X_train = train[feature_cols].fillna(0)
    y_train = train["Consumption_Scaled"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"    LR trained on {len(feature_cols)} features, {len(X_train)} rows")

    return {
        "model": model,
        "scaler_target": scaler_target,
        "scaler_weather": scaler_weather,
        "feature_cols": feature_cols,
        "history_scaled": train["Consumption_Scaled"].values.tolist(),
    }

# Train Prophet model for a given client. 
def train_prophet(client_id, df_all):
    client_df = df_all[df_all["ClientID"]==client_id].copy()
    client_df = client_df.rename(columns={"Date":"ds","Consumption":"y"})
    client_df = client_df.sort_values("ds").reset_index(drop=True)
    client_df = client_df.dropna(subset=["Lag_15min","Lag_24h","Rolling_Mean_4h"]).reset_index(drop=True)

    train_df = client_df.iloc[:-FORECAST_HORIZON].copy()
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays(country_name="PT")
    m.add_regressor("Temp_National_Avg")
    m.add_regressor("Lag_15min")
    m.add_regressor("Lag_24h")
    m.add_regressor("Rolling_Mean_4h")
    m.fit(train_df[["ds","y","Temp_National_Avg","Lag_15min","Lag_24h","Rolling_Mean_4h"]])

    return {
        "model": m,
        "history_y": train_df["y"].values.tolist(),
        "last_ds": train_df["ds"].iloc[-1],
    }

# Train SARIMAX model for a given client.
def train_sarimax(client_id, df_all):
    df_c = df_all[df_all["ClientID"]==client_id].copy()
    df_c = df_c.set_index("Date")
    df_c.index = pd.to_datetime(df_c.index)
    df_c = df_c.sort_index()

    # Forward fill to ensure continuous 15-min intervals
    full_index = pd.date_range(df_c.index.min(), df_c.index.max(), freq="15min")
    df_c = df_c.reindex(full_index).ffill()

    y_full    = df_c["Consumption"]
    exog_full = df_c[EXOG_COLS_SARIMAX].astype(float)


    # Use the last 2 weeks of data for training
    n           = len(y_full)
    test_start  = n - FORECAST_HORIZON
    train_start = test_start - (SARIMAX_TRAIN_WEEKS * 7 * 96)

    if train_start < 0:
        raise ValueError(f"Client {client_id}: not enough data.")

    y_train    = y_full.iloc[train_start:test_start]
    exog_train = exog_full.iloc[train_start:test_start]

    print(f"    SARIMAX fitting {len(y_train)} rows (takes ~1-2 min)...")

    result = SARIMAX(
        y_train, exog=exog_train,
        order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=50)

    return {
        "result": result,
        "last_train_index": y_train.index[-1],
        "exog_cols": EXOG_COLS_SARIMAX,
        "freq": "15min",
    }


# Save trained model dictionary as a pickle file.
def save_artifact(client_id, model_name, artifact):
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_{client_id}.pkl")
    joblib.dump(artifact, path)
    return path

# Check if artifact exists for a given client and model. 
def artifact_exists(client_id, model_name):
    return os.path.exists(os.path.join(ARTIFACTS_DIR, f"{model_name}_{client_id}.pkl"))

# Iterates over all clients and trains/saves models. 
def train_all(clients, df, force=False):
    results = {"lr":[],"prophet":[],"sarimax":[],"failed":[]}

    for client_id in clients:
        print(f"\n{'─'*50}")
        print(f"  Client: {client_id}")

        if force or not artifact_exists(client_id,"lr"):
            try:
                save_artifact(client_id,"lr",train_linear_regression(client_id,df))
                results["lr"].append(client_id)
                print(f"  [LR]      saved")
            except Exception as e:
                print(f"  [LR]      FAILED: {e}")
                results["failed"].append(("lr",client_id,str(e)))
        else:
            print(f"  [LR]      already exists (skip)")

        if force or not artifact_exists(client_id,"prophet"):
            try:
                save_artifact(client_id,"prophet",train_prophet(client_id,df))
                results["prophet"].append(client_id)
                print(f"  [Prophet] saved")
            except Exception as e:
                print(f"  [Prophet] FAILED: {e}")
                results["failed"].append(("prophet",client_id,str(e)))
        else:
            print(f"  [Prophet] already exists (skip)")

        if force or not artifact_exists(client_id,"sarimax"):
            try:
                save_artifact(client_id,"sarimax",train_sarimax(client_id,df))
                results["sarimax"].append(client_id)
                print(f"  [SARIMAX] saved")
            except Exception as e:
                print(f"  [SARIMAX] FAILED: {e}")
                results["failed"].append(("sarimax",client_id,str(e)))
        else:
            print(f"  [SARIMAX] already exists (skip)")

    print(f"\n{'='*50}")
    print(f"Training complete.")
    print(f"  LR:      {len(results['lr'])} clients saved")
    print(f"  Prophet: {len(results['prophet'])} clients saved")
    print(f"  SARIMAX: {len(results['sarimax'])} clients saved")
    if results["failed"]:
        print(f"  Failed:  {len(results['failed'])}")
        for item in results["failed"]:
            print(f"    {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", nargs="+")
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data",  default=PARQUET_PATH)
    args = parser.parse_args()

    df          = load_data(args.data)
    all_clients = sorted(df["ClientID"].unique().tolist())

    if args.clients:
        clients_to_train = args.clients
    elif args.all:
        clients_to_train = all_clients
    else:
        random.seed(42)
        clients_to_train = random.sample(all_clients, 30)
        print("No --clients or --all specified. Using default 30-client sample (seed=42).")

    train_all(clients_to_train, df, force=args.force)