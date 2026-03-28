import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# Ensure project root is in sys.path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods

def load_processed_data(file_path):
    print("Loading processed data...")
    return pd.read_parquet(file_path)

def preprocess_and_split(df_long):
    print("Feature Engineering and Train/Test Split...")
    df_long['Is_Weekend'] = df_long['Is_Weekend'].astype(int)
    df_long['Is_Holiday'] = df_long['Is_Holiday'].astype(int)

    df_model = pd.get_dummies(df_long, columns=['Hour', 'Weekday', 'Month', 'Consumer_Category'], drop_first=True)

    train = df_model[df_model['Date'].dt.year < 2014].copy()
    test  = df_model[df_model['Date'].dt.year >= 2014].copy()

    weather_cols = ['HDH', 'CDH', 'HDH_lag24h', 'CDH_lag24h', 'HDH_anomaly', 'CDH_anomaly']
    scaler_weather = StandardScaler()
    train[weather_cols] = scaler_weather.fit_transform(train[weather_cols])
    test[weather_cols]  = scaler_weather.transform(test[weather_cols])

    train = train.sort_values(by=['ClientID', 'Date'])
    test  = test.sort_values(by=['ClientID', 'Date'])

    for col in ['Lag_15min_Scaled', 'Lag_1h_Scaled', 'Lag_24h_Scaled', 'Lag_1week_Scaled', 'Rolling_Mean_4h_Scaled']:
        train[col] = np.nan
        test[col]  = np.nan

    client_scalers = {}

    for client in tqdm(df_long['ClientID'].unique(), desc="Scaling Clients"):
        scaler = StandardScaler()
        train_mask = train['ClientID'] == client
        test_mask  = test['ClientID'] == client

        if not train_mask.any():
            print(f"Warning: Client {client} has no data in the train set. Skipping...")
            continue

        train.loc[train_mask, 'Consumption_Scaled'] = scaler.fit_transform(
            train.loc[train_mask, 'Consumption'].values.reshape(-1, 1)
        ).flatten()

        train.loc[train_mask, 'Lag_15min_Scaled'] = scaler.transform(train.loc[train_mask, 'Lag_15min'].values.reshape(-1, 1)).flatten()
        train.loc[train_mask, 'Lag_1h_Scaled'] = scaler.transform(train.loc[train_mask, 'Lag_1h'].values.reshape(-1, 1)).flatten()
        train.loc[train_mask, 'Lag_24h_Scaled'] = scaler.transform(train.loc[train_mask, 'Lag_24h'].values.reshape(-1, 1)).flatten()
        train.loc[train_mask, 'Lag_1week_Scaled'] = scaler.transform(train.loc[train_mask, 'Lag_1week'].values.reshape(-1, 1)).flatten()
        train.loc[train_mask, 'Rolling_Mean_4h_Scaled'] = scaler.transform(train.loc[train_mask, 'Rolling_Mean_4h'].values.reshape(-1, 1)).flatten()

        if test_mask.any():
            test.loc[test_mask, 'Consumption_Scaled'] = scaler.transform(test.loc[test_mask, 'Consumption'].values.reshape(-1, 1)).flatten()
            test.loc[test_mask, 'Lag_15min_Scaled'] = scaler.transform(test.loc[test_mask, 'Lag_15min'].values.reshape(-1, 1)).flatten()
            test.loc[test_mask, 'Lag_1h_Scaled'] = scaler.transform(test.loc[test_mask, 'Lag_1h'].values.reshape(-1, 1)).flatten()
            test.loc[test_mask, 'Lag_24h_Scaled'] = scaler.transform(test.loc[test_mask, 'Lag_24h'].values.reshape(-1, 1)).flatten()
            test.loc[test_mask, 'Lag_1week_Scaled'] = scaler.transform(test.loc[test_mask, 'Lag_1week'].values.reshape(-1, 1)).flatten()
            test.loc[test_mask, 'Rolling_Mean_4h_Scaled'] = scaler.transform(test.loc[test_mask, 'Rolling_Mean_4h'].values.reshape(-1, 1)).flatten()

        client_scalers[client] = scaler

    train = train.dropna(subset=['Consumption_Scaled', 'Lag_15min_Scaled', 'Lag_1h_Scaled', 'Lag_24h_Scaled', 'Lag_1week_Scaled', 'Rolling_Mean_4h_Scaled'])

    cols_to_drop = ['Date', 'ClientID', 'Consumption', 'Consumption_Scaled', 'Lag_15min', 'Lag_1h', 'Lag_24h', 'Lag_1week', 'Rolling_Mean_4h', 'Lag_15min_Scaled', 'Lag_1h_Scaled', 'Rolling_Mean_4h_Scaled', 'Temp_National_Avg']

    cols_to_drop_train = [c for c in cols_to_drop if c in train.columns]
    
    X_train = train.drop(columns=cols_to_drop_train)
    y_train = train['Consumption_Scaled']

    test = test.sort_values(by=['ClientID', 'Date'])
    X_test = test.drop(columns=cols_to_drop_train)

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")

    feature_cols = X_train.drop(columns=['Cluster'], errors='ignore').columns.tolist()

    return train, test, X_train, y_train, X_test, client_scalers, scaler_weather, feature_cols

def train_models(X_train, y_train, train):
    print("Training Linear Regression models per cluster...")
    cluster_models = {}
    unique_clusters = train['Cluster'].unique()

    for cluster_id in sorted(unique_clusters):
        cluster_mask = train['Cluster'] == cluster_id
        X_train_cluster = X_train[cluster_mask].drop(columns=['Cluster'], errors='ignore')
        y_train_cluster = y_train[cluster_mask]

        model = LinearRegression()
        model.fit(X_train_cluster.values, y_train_cluster)
        cluster_models[cluster_id] = model
        print(f" - Model for Cluster {int(cluster_id)} trained on {len(X_train_cluster)} historical rows.")

    return cluster_models

def predict_models(cluster_models, test, X_test, client_scalers):
    print("Predicting on Test Set...")
    test['Predicted_Consumption_Scaled'] = np.nan

    for cluster_id, model in cluster_models.items():
        cluster_mask = test['Cluster'] == cluster_id
        X_test_cluster = X_test[cluster_mask].drop(columns=['Cluster'], errors='ignore')
        
        if len(X_test_cluster) > 0:
            preds = model.predict(X_test_cluster.values)
            test.loc[cluster_mask, 'Predicted_Consumption_Scaled'] = preds

    print("Applying physical constraints (Capping at 0 kW)...")
    for client in test['ClientID'].unique():
        if client in client_scalers:
            min_scaled_val = client_scalers[client].transform([[0.0]])[0][0]
            c_mask = test['ClientID'] == client
            test.loc[c_mask, 'Predicted_Consumption_Scaled'] = np.maximum(
                test.loc[c_mask, 'Predicted_Consumption_Scaled'], 
                min_scaled_val
            )

    print("Predictions Complete!")
    return test

def evaluate_models(test, client_scalers):
    """
    Inverse-transform scaled predictions back to kW for every client, then
    compute cluster-level and global MAPE / WMAPE via the shared evaluation module.
    """
    print("\nEvaluating model (raw kW)...")
    for client in test['ClientID'].unique():
        if client not in client_scalers:
            continue

        client_mask = test['ClientID'] == client
        client_data = test[client_mask].copy()

        valid = client_data['Consumption'].notna() & client_data['Predicted_Consumption_Scaled'].notna()
        if valid.sum() == 0:
            continue

        y_true_kw = client_data.loc[valid, 'Consumption'].values
        scaler = client_scalers[client]
        y_pred_kw = scaler.inverse_transform(
            client_data.loc[valid, 'Predicted_Consumption_Scaled'].values.reshape(-1, 1)
        ).flatten()

        y_pred_kw = np.maximum(y_pred_kw, 0)
        test.loc[client_data.index[valid], 'Actual_kW'] = y_true_kw
        test.loc[client_data.index[valid], 'Predicted_kW'] = y_pred_kw

    cluster_eval = (
        test.dropna(subset=['Actual_kW', 'Predicted_kW'])
        .groupby(['Cluster', 'Date'], observed=True)[['Actual_kW', 'Predicted_kW']]
        .sum()
        .reset_index()
    )

    summary = compute_cluster_metrics(cluster_eval)

    return cluster_eval, summary


def save_cluster_artifacts(cluster_models, client_scalers, scaler_weather, feature_cols, client_clusters, artifacts_dir="../agent/artifacts"):
    print(f"Saving Cluster Linear Regression artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    artifact = {
        "cluster_models": cluster_models,
        "client_scalers": client_scalers,
        "scaler_weather": scaler_weather,
        "feature_cols": list(feature_cols),
        "client_clusters": {k: v for k, v in client_clusters.items()}
    }
    
    path = os.path.join(artifacts_dir, "lr_cluster_models.pkl")
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")

def run_linear_regression_pipeline(file_path, plot=False):
    """
    Complete pipeline to load data, train models, predict, evaluate, and visualize results.
    """
    df_long = load_processed_data(file_path)
    train, test, X_train, y_train, X_test, client_scalers, scaler_weather, feature_cols = preprocess_and_split(df_long)
    cluster_models = train_models(X_train, y_train, train)
    test = predict_models(cluster_models, test, X_test, client_scalers)
    cluster_eval, summary = evaluate_models(test, client_scalers)
    
    client_clusters = df_long.drop_duplicates(subset=['ClientID']).set_index('ClientID')['Cluster'].to_dict()
    save_cluster_artifacts(cluster_models, client_scalers, scaler_weather, feature_cols, client_clusters, artifacts_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'agent', 'artifacts'))
    
    if plot:
        plot_cluster_portfolio(cluster_eval, summary)
        analyze_time_periods(test)
    
    return cluster_models, test, cluster_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "Datasets", "processed_electricity_data.parquet")
    run_linear_regression_pipeline(DATA_PATH, plot=False)