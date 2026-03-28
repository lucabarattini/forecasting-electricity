import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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

    weather_cols = ['HDH', 'CDH']
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

    cols_to_drop = ['Date', 'ClientID', 'DayMonth', 'Consumption', 'Consumption_Scaled',
                    'Lag_15min', 'Lag_1h', 'Lag_24h', 'Lag_1week', 'Rolling_Mean_4h',
                    'Lag_15min_Scaled', 'Lag_1h_Scaled', 'Rolling_Mean_4h_Scaled',
                    'Temp_National_Avg', 'HDH_lag24h', 'CDH_lag24h', 'HDH_anomaly', 'CDH_anomaly']

    cols_to_drop_train = [c for c in cols_to_drop if c in train.columns]
    
    X_train = train.drop(columns=cols_to_drop_train)
    y_train = train['Consumption_Scaled']

    test = test.sort_values(by=['ClientID', 'Date'])
    X_test = test.drop(columns=cols_to_drop_train)

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")

    return train, test, X_train, y_train, X_test, client_scalers

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
    print("Predicting on Test Set (Vectorized Day-Ahead)...")
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

    cluster_eval = test.dropna(subset=['Actual_kW', 'Predicted_kW']).groupby(['Cluster', 'Date'], observed=True)[['Actual_kW', 'Predicted_kW']].sum().reset_index()
    cluster_eval['Abs_Error'] = np.abs(cluster_eval['Actual_kW'] - cluster_eval['Predicted_kW'])

    mask_mape = cluster_eval['Actual_kW'] > 0.1
    cluster_eval.loc[mask_mape, 'Perc_Error'] = (cluster_eval['Abs_Error'] / cluster_eval['Actual_kW']) * 100

    print("\n--- LINEAR REGRESSION PERFORMANCE BY CLUSTER (BUSINESS ORIENTED) ---\n")
    summary = cluster_eval.groupby('Cluster', observed=True).agg(
        Portfolio_MAPE=('Perc_Error', 'mean'),
        Total_Abs_Error=('Abs_Error', 'sum'),
        Total_Actual=('Actual_kW', 'sum')
    )
    summary['Portfolio_WMAPE'] = (summary['Total_Abs_Error'] / summary['Total_Actual']) * 100
    summary = summary.drop(columns=['Total_Abs_Error', 'Total_Actual']).round(2)
    print(summary)

    global_wmape = (cluster_eval['Abs_Error'].sum() / cluster_eval['Actual_kW'].sum()) * 100
    global_mape = cluster_eval['Perc_Error'].mean()
    print(f"\nGlobal Portfolio MAPE: {global_mape:.2f}%")
    print(f"Global Portfolio WMAPE: {global_wmape:.2f}%")

    return cluster_eval, summary

def plot_cluster_portfolio(cluster_eval, summary):
    print("Generating Cluster Portfolio visualizations...")
    unique_clusters = sorted(cluster_eval['Cluster'].unique())
    fig, axes = plt.subplots(len(unique_clusters), 1, figsize=(15, 5 * len(unique_clusters)))

    if len(unique_clusters) == 1:
        axes = [axes]

    for idx, cluster_id in enumerate(unique_clusters):
        ax = axes[idx]
        c_plot = cluster_eval[cluster_eval['Cluster'] == cluster_id].sort_values('Date')
        c_mape = summary.loc[cluster_id, 'Portfolio_MAPE']
        c_wmape = summary.loc[cluster_id, 'Portfolio_WMAPE']
        plot_slice = -1344 
        
        ax.plot(c_plot['Date'].values[plot_slice:], c_plot['Actual_kW'].values[plot_slice:],
                label='Actual Portfolio Load', color='blue', alpha=0.6, linewidth=2)
        ax.plot(c_plot['Date'].values[plot_slice:], c_plot['Predicted_kW'].values[plot_slice:],
                label='LR Prediction (Day-Ahead)', color='red', linestyle='--', alpha=0.9, linewidth=1.5)

        ax.set_title(f'Cluster {cluster_id} Portfolio — (MAPE: {c_mape:.2f}% | WMAPE: {c_wmape:.2f}%)', fontsize=14)
        ax.set_ylabel('Total Consumption (kW)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

def analyze_time_periods(test):
    print("Preparing data for Time Period Analysis...")
    df_eval = test.dropna(subset=['Actual_kW', 'Predicted_kW']).copy()
    df_eval = df_eval.sort_values('Date')

    _, bin_edges = pd.cut(df_eval['Date'], bins=4, retbins=True)
    bin_edges = pd.to_datetime(bin_edges)
    dynamic_labels = [f"{bin_edges[i].strftime('%b %d')} to {bin_edges[i+1].strftime('%b %d')}" for i in range(4)]

    df_eval['Time_Period'] = pd.cut(df_eval['Date'], bins=4, labels=dynamic_labels)

    portfolio_ts = df_eval.groupby(['Date', 'Time_Period'], observed=True)[['Actual_kW', 'Predicted_kW']].sum().reset_index()
    portfolio_ts['Abs_Error'] = np.abs(portfolio_ts['Actual_kW'] - portfolio_ts['Predicted_kW'])
    portfolio_ts['APE'] = (portfolio_ts['Abs_Error'] / portfolio_ts['Actual_kW']) * 100

    print("\n--- PORTFOLIO PERFORMANCE BY TIME PERIOD (Aggregated) ---")
    portfolio_wmape = portfolio_ts.groupby('Time_Period', observed=True).apply(
        lambda x: (x['Abs_Error'].sum() / x['Actual_kW'].sum()) * 100
    )

    for period, wmape_val in portfolio_wmape.items():
        avg_mape = portfolio_ts[portfolio_ts['Time_Period'] == period]['APE'].mean()
        print(f"{period}:   WMAPE = {wmape_val:.2f}% | MAPE = {avg_mape:.2f}%")

    df_eval['Abs_Error'] = np.abs(df_eval['Actual_kW'] - df_eval['Predicted_kW'])
    mask_ape = df_eval['Actual_kW'] > 0.1
    df_ape = df_eval[mask_ape].copy()
    df_ape['APE'] = (df_ape['Abs_Error'] / df_ape['Actual_kW']) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df_ape, x='Time_Period', y='APE', ax=axes[0], showfliers=False)
    axes[0].set_title('Individual Client Percentage Error Spread (APE)', fontsize=14)
    axes[0].set_ylabel('APE (%)')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)

    sns.boxplot(data=df_eval, x='Time_Period', y='Abs_Error', ax=axes[1], showfliers=False)
    axes[1].set_title('Individual Client Volume Error Spread (kW)', fontsize=14)
    axes[1].set_ylabel('Absolute Error (kW)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.show()

def run_linear_regression_pipeline(file_path):
    """
    Complete pipeline to load data, train models, predict, evaluate, and visualize results.
    """
    df_long = load_processed_data(file_path)
    train, test, X_train, y_train, X_test, client_scalers = preprocess_and_split(df_long)
    cluster_models = train_models(X_train, y_train, train)
    test = predict_models(cluster_models, test, X_test, client_scalers)
    cluster_eval, summary = evaluate_models(test, client_scalers)
    
    plot_cluster_portfolio(cluster_eval, summary)
    analyze_time_periods(test)
    
    return cluster_models, test, cluster_eval, summary