"""
transformer_model.py
--------------------
Modularized implementation of the Non-Stationary Transformer (NeurIPS 2022).
One model per behavioral cluster, with hourly aggregation and client-level unscaling.

Key Components:
1. PyTorch Architecture: NonStationaryTransformer with DSAttention.
2. Pipeline: Preprocessing, Cluster Training, Rolling Forecast, Safe Merge, Evaluation.
"""

import os
import sys
import math
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Environment setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods
# =============================================================
# DEVICE DETECTION
# =============================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# =============================================================
# PYTORCH MODEL CLASSES (Non-Stationary Transformer)
# =============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, n_time_features=2, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.temporal_embedding = nn.Linear(n_time_features, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, x_mark):
        out = self.value_embedding(x) + self.position_encoding(x) + self.temporal_embedding(x_mark)
        return self.dropout(out)

class DSAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        
        tau   = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = DSAttention(mask_flag=False, dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection   = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, _ = self.attention(x, x, x, attn_mask, tau, delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, attn_mask, tau, delta)
        if self.norm:
            x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention  = AttentionLayer(d_model, n_heads, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, x_mask, tau, None)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, cross_mask, tau, delta)[0])
        y = x = self.norm2(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask, tau, delta)
        if self.norm:
            x = self.norm(x)
        if self.projection:
            x = self.projection(x)
        return x

class Projector(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims, output_dim, kernel_size=3):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(seq_len, 1, kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, x, stats):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)
        return self.backbone(x)

class NonStationaryTransformer(nn.Module):
    def __init__(self, enc_in, c_out, seq_len, label_len, pred_len,
                 d_model=128, n_heads=4, e_layers=2, d_layers=1,
                 d_ff=256, dropout=0.1, p_hidden_dims=[64], n_time_features=2):
        super().__init__()
        self.pred_len  = pred_len
        self.seq_len   = seq_len
        self.label_len = label_len
        self.enc_embedding = DataEmbedding(enc_in, d_model, n_time_features, dropout)
        self.dec_embedding = DataEmbedding(enc_in, d_model, n_time_features, dropout)
        self.encoder = Encoder(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out)
        )
        self.tau_learner   = Projector(enc_in, seq_len, p_hidden_dims, output_dim=1)
        self.delta_learner = Projector(enc_in, seq_len, p_hidden_dims, output_dim=seq_len)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([
            x_enc[:, -self.label_len:, :],
            torch.zeros_like(x_dec[:, -self.pred_len:, :])
        ], dim=1).to(x_enc.device)
        tau   = self.tau_learner(x_raw, std_enc).exp()
        delta = self.delta_learner(x_raw, mean_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, tau=tau, delta=delta)
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, tau=tau, delta=delta)
        dec_out = dec_out * std_enc[:, :, :1] + mean_enc[:, :, :1]
        return dec_out[:, -self.pred_len:, :]

# =============================================================
# DATASET HANDLING
# =============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, label_len: int, pred_len: int):
        self.data     = data.astype(np.float32)
        self.seq_len  = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self._time_features(s_begin, s_end)
        seq_y_mark = self._time_features(r_begin, r_end)
        return (torch.tensor(seq_x), torch.tensor(seq_y),
                torch.tensor(seq_x_mark), torch.tensor(seq_y_mark))
    
    def _time_features(self, start, end):
        hours = np.arange(start, end) % 24 / 23.0 - 0.5
        days  = (np.arange(start, end) // 24) % 7 / 6.0 - 0.5
        return np.stack([hours, days], axis=-1).astype(np.float32)

# =============================================================
# PIPELINE FUNCTIONS
# =============================================================

def preprocess_and_split(df_long, mode='day_ahead'):
    print(f"Preparing train/test split and scaling (Mode: {mode.upper()})...")
    
    train_raw = df_long[df_long['Date'].dt.year < 2014].copy()
    test_raw  = df_long[df_long['Date'].dt.year >= 2014].copy()

    calendar_cols = ['Is_Weekend', 'Is_Holiday']
    calendar_cols = [c for c in calendar_cols if c in df_long.columns]
    
    # 2. Weather Scaling
    weather_cols = ['HDH', 'CDH', 'HDH_lag24h', 'CDH_lag24h', 'HDH_anomaly', 'CDH_anomaly']
    weather_cols = [c for c in weather_cols if c in df_long.columns]
    
    scaler_weather = StandardScaler()
    train_raw[weather_cols] = scaler_weather.fit_transform(train_raw[weather_cols].astype(np.float32))
    test_raw[weather_cols]  = scaler_weather.transform(test_raw[weather_cols].astype(np.float32))
    
    regressors = ['Consumption'] + weather_cols + calendar_cols
    
    client_scalers = {}
    train_raw['Consumption_Scaled'] = np.nan
    test_raw['Consumption_Scaled'] = np.nan

    for client in tqdm(df_long['ClientID'].unique(), desc="Scaling Clients"):
        scaler = StandardScaler()
        train_mask = train_raw['ClientID'] == client
        test_mask  = test_raw['ClientID'] == client
        if not train_mask.any(): continue
            
        train_raw.loc[train_mask, 'Consumption_Scaled'] = scaler.fit_transform(
            train_raw.loc[train_mask, 'Consumption'].values.reshape(-1, 1)
        ).flatten()
        
        if test_mask.any():
            test_raw.loc[test_mask, 'Consumption_Scaled'] = scaler.transform(
                test_raw.loc[test_mask, 'Consumption'].values.reshape(-1, 1)
            ).flatten()
        client_scalers[client] = scaler

    print("Aggregating data by Cluster for Transformer training...")
    # Transformer targets Scaled consumption
    agg_dict = {'Consumption_Scaled': 'mean'}
    for col in weather_cols:
        agg_dict[col] = 'first'
    for col in calendar_cols:
        agg_dict[col] = 'max'  # Keep binary flags 0 or 1 during resampling
        
    train_agg = train_raw.groupby(['Cluster', 'Date'], observed=True).agg(agg_dict).reset_index()
    test_agg  = test_raw.groupby(['Cluster', 'Date'], observed=True).agg(agg_dict).reset_index()
    
    train_agg = train_agg.sort_values(['Cluster', 'Date'])
    test_agg  = test_agg.sort_values(['Cluster', 'Date'])
    
    # We rename Consumption_Scaled to Consumption for the model to treat it as target
    train_agg = train_agg.rename(columns={'Consumption_Scaled': 'Consumption'})
    test_agg = test_agg.rename(columns={'Consumption_Scaled': 'Consumption'})
    
    return train_agg, test_agg, test_raw, client_scalers, scaler_weather, regressors

def train_models(train_agg, regressors, params=None):
    device = get_device()
    p = params or {
        'SEQ_LEN': 168, 'LABEL_LEN': 24, 'PRED_LEN': 24,
        'D_MODEL': 128, 'N_HEADS': 4, 'E_LAYERS': 2, 'D_LAYERS': 1,
        'D_FF': 256, 'DROPOUT': 0.1, 'BATCH_SIZE': 32,
        'LR': 100e-6, 'EPOCHS': 10, 'PATIENCE': 3
    }
    
    cluster_models = {}
    unique_clusters = sorted(train_agg['Cluster'].unique())
    
    for cluster_id in unique_clusters:
        print(f"\nTraining Cluster {cluster_id}")
        df_c = train_agg[train_agg['Cluster'] == cluster_id].set_index('Date').sort_index()
        df_c = df_c[regressors].resample('1h').mean().ffill().dropna()
        
        val_size = max(int(len(df_c) * 0.1), p['SEQ_LEN'] + p['PRED_LEN'] + 1)
        train_data_np = df_c.values[:-val_size]
        val_data_np   = df_c.values[-val_size:]
        
        train_ds = TimeSeriesDataset(train_data_np, p['SEQ_LEN'], p['LABEL_LEN'], p['PRED_LEN'])
        val_ds   = TimeSeriesDataset(val_data_np, p['SEQ_LEN'], p['LABEL_LEN'], p['PRED_LEN'])
        
        train_loader = DataLoader(train_ds, batch_size=p['BATCH_SIZE'], shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=p['BATCH_SIZE'], shuffle=False, drop_last=False)
        
        model = NonStationaryTransformer(
            enc_in=len(regressors), c_out=1,
            seq_len=p['SEQ_LEN'], label_len=p['LABEL_LEN'], pred_len=p['PRED_LEN'],
            d_model=p['D_MODEL'], n_heads=p['N_HEADS'], e_layers=p['E_LAYERS'],
            d_layers=p['D_LAYERS'], d_ff=p['D_FF'], dropout=p['DROPOUT']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=p['LR'])
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(p['EPOCHS']):
            model.train()
            train_losses = []
            for b_x, b_y, b_x_mark, b_y_mark in train_loader:
                b_x, b_y, b_x_mark, b_y_mark = b_x.to(device), b_y.to(device), b_x_mark.to(device), b_y_mark.to(device)
                outputs = model(b_x, b_x_mark, b_y, b_y_mark)
                loss = criterion(outputs, b_y[:, -p['PRED_LEN']:, :1])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for b_x, b_y, b_x_mark, b_y_mark in val_loader:
                    b_x, b_y, b_x_mark, b_y_mark = b_x.to(device), b_y.to(device), b_x_mark.to(device), b_y_mark.to(device)
                    outputs = model(b_x, b_x_mark, b_y, b_y_mark)
                    val_losses.append(criterion(outputs, b_y[:, -p['PRED_LEN']:, :1]).item())
            
            avg_val = np.mean(val_losses)
            print(f"  Epoch {epoch+1} | Train: {np.mean(train_losses):.5f} | Val: {avg_val:.5f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= p['PATIENCE']: break
        
        model.load_state_dict(best_state)
        cluster_models[cluster_id] = model
        
        # Memory Cleanup after each cluster
        del train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
    return cluster_models

def predict_models(cluster_models, train_agg, test_agg, test_raw, client_scalers, regressors, params=None):
    device = get_device()
    p = params or {'SEQ_LEN': 168, 'LABEL_LEN': 24, 'PRED_LEN': 24}
    all_cluster_forecasts = []
    
    for cluster_id, model in cluster_models.items():
        print(f"Generating forecasts for Cluster {int(cluster_id)}...")
        model.eval()
        
        df_train_c = train_agg[train_agg['Cluster'] == cluster_id].set_index('Date').sort_index()
        df_train_c = df_train_c[regressors].resample('1h').mean().ffill().dropna()
        
        df_test_c = test_agg[test_agg['Cluster'] == cluster_id].set_index('Date').sort_index()
        df_test_c = df_test_c[regressors].resample('1h').mean().ffill().dropna()

        full_scaled = np.concatenate([df_train_c.values, df_test_c.values], axis=0)
        test_start_idx = len(df_train_c)
        n_test = len(df_test_c)
        
        # Initialize dataset once per cluster
        mock_ds = TimeSeriesDataset(full_scaled, p['SEQ_LEN'], p['LABEL_LEN'], p['PRED_LEN'])
        
        all_preds = []
        with torch.no_grad():
            for day_offset in range(0, n_test, p['PRED_LEN']):
                window_end = test_start_idx + day_offset
                window_start = window_end - p['SEQ_LEN']
                if window_start < 0 or window_end + p['PRED_LEN'] > len(full_scaled): break
                
                enc_x = torch.tensor(full_scaled[window_start:window_end], dtype=torch.float32).unsqueeze(0).to(device)
                dec_x = torch.tensor(full_scaled[window_end - p['LABEL_LEN']:window_end + p['PRED_LEN']], dtype=torch.float32).unsqueeze(0).to(device)
                
                enc_mark = torch.tensor(mock_ds._time_features(window_start, window_end), dtype=torch.float32).unsqueeze(0).to(device)
                dec_mark = torch.tensor(mock_ds._time_features(window_end - p['LABEL_LEN'], window_end + p['PRED_LEN']), dtype=torch.float32).unsqueeze(0).to(device)
                
                output = model(enc_x, enc_mark, dec_x, dec_mark)
                all_preds.append(output.squeeze(0).cpu().numpy())

        if all_preds:
            forecast_scaled = np.concatenate(all_preds, axis=0)[:n_test]
            
            start_date = df_test_c.index[0]
            safe_dates = pd.date_range(start=start_date, periods=len(forecast_scaled), freq='1h')
            
            fcst_hourly = pd.DataFrame({
                'Cluster': cluster_id,
                'Date': safe_dates,
                'Predicted_Consumption_Scaled': forecast_scaled[:, 0]
            }).set_index('Date')
            
            # Resample a 15 min per coprire l'intero segmento
            fcst_15min = fcst_hourly.resample('15min').ffill().reset_index()
            all_cluster_forecasts.append(fcst_15min)
            
    global_forecasts = pd.concat(all_cluster_forecasts, ignore_index=True)
    
    # Allineamento tipi per il merge (Cluster deve essere dello stesso tipo)
    test_raw['Cluster'] = test_raw['Cluster'].astype(float)
    global_forecasts['Cluster'] = global_forecasts['Cluster'].astype(float)
    
    test_raw = test_raw.drop(columns=['Predicted_Consumption_Scaled'], errors='ignore')
    test_raw = test_raw.merge(global_forecasts, on=['Cluster', 'Date'], how='left')
    
    test_raw['Predicted_kW'] = np.nan
    test_raw['Actual_kW'] = test_raw['Consumption']
    
    for client in tqdm(test_raw['ClientID'].unique(), desc="Un-scaling Clients"):
        mask = test_raw['ClientID'] == client
        if mask.any() and client in client_scalers:
            scaler = client_scalers[client]
            preds_scaled = test_raw.loc[mask, 'Predicted_Consumption_Scaled'].values.reshape(-1, 1)
            
            # CORREZIONE: Gestione flessibile dei NaN - non scartiamo l'intero cliente
            nan_mask = np.isnan(preds_scaled).flatten()
            if not nan_mask.all():
                # Prepariamo un array di output
                unscaled_full = np.full(preds_scaled.shape, np.nan)
                valid_idx = ~nan_mask
                
                # Applichiamo inverse_transform solo sui dati validi
                unscaled_valid = scaler.inverse_transform(preds_scaled[valid_idx]).flatten()
                unscaled_full[valid_idx, 0] = unscaled_valid
                
                test_raw.loc[mask, 'Predicted_kW'] = np.maximum(unscaled_full.flatten(), 0)
                
    return test_raw

def evaluate_models(test_raw):
    print("\nEvaluating Portfolio Performance...")

    valid_data = test_raw.dropna(subset=['Actual_kW', 'Predicted_kW'])
    if valid_data.empty:
        print("WARNING: No valid predictions found to evaluate. Portfolio metrics will be empty.")
        return pd.DataFrame(), pd.DataFrame()
        
    portfolio_eval = (
        valid_data
        .groupby(['Cluster', 'Date'], observed=True)[['Actual_kW', 'Predicted_kW']]
        .sum()
        .reset_index()
    )

    summary = compute_cluster_metrics(portfolio_eval)

    return portfolio_eval, summary

def save_transformer_artifacts(cluster_models, client_scalers, scaler_weather, regressors, mode, artifacts_dir=None):
    if artifacts_dir is None:
        artifacts_dir = os.path.join(PROJECT_ROOT, 'agent', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Bundle PyTorch state dicts instead of the whole model object to avoid pickling issues
    cluster_states = {c: model.state_dict() for c, model in cluster_models.items()}
    
    artifact = {
        "cluster_states": cluster_states,
        "client_scalers": client_scalers,
        "scaler_weather": scaler_weather,
        "regressors": regressors,
        "mode": mode
    }
    path = os.path.join(artifacts_dir, f"nst_cluster_{mode}.pkl")
    joblib.dump(artifact, path)
    print(f"\nTransformer artifacts saved to: {path}")

def run_transformer_pipeline(file_path, mode='day_ahead', plot=False):
    df_long = pd.read_parquet(file_path)
    train_agg, test_agg, test_raw, client_scalers, scaler_weather, regressors = preprocess_and_split(df_long, mode=mode)
    
    cluster_models = train_models(train_agg, regressors)
    test_raw = predict_models(cluster_models, train_agg, test_agg, test_raw, client_scalers, regressors)
    portfolio_eval, summary = evaluate_models(test_raw)
    
    save_transformer_artifacts(cluster_models, client_scalers, scaler_weather, regressors, mode)
    
    if plot:
        plot_cluster_portfolio(portfolio_eval, summary, model_label="NS-Transformer")
        analyze_time_periods(test_raw)
        
    return cluster_models, test_raw, portfolio_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "Datasets", "processed_electricity_data.parquet")
    run_transformer_pipeline(DATA_PATH, plot=True)
