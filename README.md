<div align="center">

<table border="0" cellspacing="0" cellpadding="12">
<tr>
  <td align="center" width="50%">
    <img src="Images%20%26%20Videos/Logo.png" width="100%" alt="Logo">
  </td>
  <td align="center" width="50%">
    <img src="Images%20%26%20Videos/Team%20-%20Thumbnail.png" width="100%" alt="Team">
  </td>
</tr>
</table>

<br>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Forecasting-scikit--learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-green?style=for-the-badge)

</div>

IEOR 4578 — Electricity load forecasting for 370 Portuguese clients (2011–2015, 15-minute intervals). We preprocess the raw data, cluster clients by consumption behavior, then benchmark three forecasting models on the same 30 clients, same 24-hour test horizon, using MAPE/WMAPE/MAE/RMSE.

## Demo

<video src="Images%20%26%20Videos/Forecasting - Demo.mp4" controls width="100%"></video>

![Demo](<Images%20%26%20Videos/Demo%20Forecasting%20Electricity.gif>)

## Graphs

<table>
<tr>
    <td><img src="Images%20%26%20Videos/Prophet%20-%201.png" width="100%"></td>
    <td><img src="Images%20%26%20Videos/Sarimax%20-%201.png" width="100%"></td>
</tr>
<tr>
    <td align="center">Prophet</td>
    <td align="center">SARIMAX</td>
</tr>
</table>

<table>
<tr>
    <td><img src="Images%20%26%20Videos/Prophet%20-%202.png" width="100%"></td>
    <td><img src="Images%20%26%20Videos/Sarimax%20-%202.png" width="100%"></td>
</tr>
<tr>
    <td align="center">Prophet</td>
    <td align="center">SARIMAX</td>
</tr>
</table>

## Repository Structure

```
forecasting-electricity/
├── Datasets/
│   ├── Electricity Dataset.csv                 # Raw data (370 clients × 140k timestamps)
│   ├── processed_electricity_data.parquet      # Output of notebook 0
│   ├── client_clusters.csv                     # Output of notebook 0.5
│   └── client_size_categories.csv
│
├── notebooks/
│   ├── 0_data_preprocessing.ipynb
│   ├── 0.5_clustering.ipynb
│   ├── 1_linear_regression.ipynb
│   ├── 2_Prophet.ipynb
│   └── 3_sarimax.ipynb
│
├── requirements.txt
└── README.md
```

## Notebooks

**0: Preprocessing**
Reshapes raw data from wide (370 columns) to long format. Adds temporal features (hour, weekday, holiday), fetches historical weather from Open-Meteo for 4 Portuguese cities (population-weighted average), engineers HDH/CDH from an 18°C comfort threshold, creates lag and rolling features (15min, 24h, 4h rolling mean), trims leading zeros for clients that joined the grid late.

**0.5: Clustering**
K-Means (k=5) on normalized 24-hour consumption profiles. Normalization is per-client (Min-Max row-wise) so the algorithm groups by shape, not volume. Produces 5 behavioral clusters (night-shift industrial, daytime commercial, late-evening residential, siesta-split, general commercial) used to break down model performance.

**1: Linear Regression**
Baseline model. One global model trained on all 30 clients combined. Lag/rolling features are recomputed per-client on standardized consumption to prevent scale leakage. Evaluation uses recursive forecasting (model predictions feed back as lag inputs). Metrics are reported in raw kW via per-client inverse transform.

**2: Prophet**
One model per client. Meta's Prophet with Portuguese holidays, temperature regressor, and lag/rolling regressors. `daily_seasonality=False` because the lag regressors already encode the daily consumption shape — having Prophet also model it via Fourier terms was double-counting and pushing the trend component off. `changepoint_prior_scale=0.15` for more flexible trend. Predictions clipped at 0. Set `DEBUG_MODE = True` to run on 3 clients (~2 min) for fast iteration.

**3: SARIMAX**
One model per client. SARIMAX(1,0,1)(1,0,1,96) on a 4-week training window. Parameters selected manually from ACF/PACF plots. `Hour` removed from exogenous variables since the seasonal component (s=96) already captures the daily pattern and treating hour as linear (1–24) was misleading the model. Set `DEBUG_MODE = True` to run on 3 clients for fast iteration.

## A note on the test window

All three models are evaluated on the same 24-hour holdout: Dec 31, 2014 (New Year's Eve). This is an atypical day, so the reported metrics reflect performance on a single difficult day rather than general accuracy.

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in order (0 → 0.5 → 1/2/3). Notebook 0 fetches weather from the Open-Meteo API and outputs the parquet file that all model notebooks depend on.

## Evaluation

All models use the same 30 clients (`random.seed(42)`), same 96-step (24-hour) holdout, metrics in raw kW. Primary metric is **MAPE** (scale-free, comparable across clients of very different sizes). Performance is broken down by behavioral cluster for all three models.

## Forecasting Agent

We built a terminal chatbot using LangChain and a Groq-hosted LLM (Llama 3.3 70B).

The agent allows users to query electricity consumption forecasts in plain English. Given a client ID and a forecast horizon, it automatically calls the models, runs recursive predictions at 15-minute resolution starting from 2015-01-01, and returns the results for each model.

### Structure

- `agent/train_and_save.py` — trains all three models (LR, Prophet, SARIMAX) per client and saves them as artifacts to disk. Run once before using the chatbot.
- `agent/models/predict.py` — inference module that loads saved artifacts and runs predictions for a given client and horizon.
- `agent/chatbot.py` — LangChain agent with a natural language interface. Parses user messages, calls the forecasting tools, and returns results.

### Usage
```bash
cd agent
pip install -r requirements_agent.txt
python train_and_save.py --clients MT_001 MT_315
python chatbot.py
```

Example query:
```
I want the forecast for client 315 for a 24-hour horizon
```

Demo YouTube link: https://youtu.be/2cXAN4B30EQ
