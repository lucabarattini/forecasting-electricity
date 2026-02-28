<div align="center">

# ⚡️ Forecasting Electricity

<img src="/Images/Logo.png" width="670" alt="Logo">
<br>
<br>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-green?style=for-the-badge)

</div>

Welcome to our first project deliverable for the **IEOR 4578** course. 

This repository contains our evolving machine learning pipeline for forecasting electricity consumption across 370 clients. The project explores end-to-step time-series forecasting, beginning with thorough data preprocessing and feature engineering, followed by client profiling (clustering), and ultimately building predictive models, which currently include a Linear Regression baseline and an advanced Meta Prophet implementation.

## 📂 Repository Structure

```text
forecasting-electricity/
│
├── Datasets/
│   ├── Electricity Dataset.csv          # Raw electricity load dataset
│   ├── client_clusters.csv              # Clients divided by clusters
│   ├── client_size_categories.csv       # Clients divided by category
│   └── processed_electricity_data.parquet # Processed dataset in .parquet for optimized caching
│
├── notebooks/
│   ├── 0_data_preprocessing.ipynb       # Feature engineering, weather data fetching, and data reshaping
│   ├── 0.5_clustering.ipynb             # K-Means clustering for client consumption profiling
│   ├── 1_linear_regression.ipynb        # Autoregressive Linear Regression baseline model
│   └── 2_Prophet.ipynb                  # Time-series modeling using Meta's Prophet open-source library
│
├── requirements.txt                     # Repo dependencies
└── README.md                            # Project documentation
```

## 🧠 Approach and Methodology

### 1. Data Preprocessing & Feature Engineering (`0_data_preprocessing.ipynb`)
The foundation of the project involves transforming the raw "wide" dataset into a machine-learning-ready "long" format. 

Key steps include:
* **Temporal Features:** Dynamically extracting Hour, Day, Weekday, and Weekend indicators.
* **Holiday Effects:** Using `dateutil.easter` to dynamically calculate and map movable and fixed Portuguese national holidays.
* **Weather Integration:** Making API calls to *Open-Meteo* to fetch historical temperatures for major Portuguese cities (Lisbon, Porto, Faro, Evora). A population-weighted national average temperature is calculated, and Heating Degree Hours (HDH) and Cooling Degree Hours (CDH) are engineered to capture temperature-driven electricity demand.
* **Data Reshaping:** Melting the dataset from 370 individual client columns into a massive "long" format, strictly downcasting data types and exporting to a highly compressed `Parquet` format to optimize memory.
* **Segmentation:** Applying Jenks optimization to segment clients into *Light*, *Medium*, and *Heavy* consumer categories based on volume.

### 2. Client Clustering (`0.5_clustering.ipynb`)
Because absolute consumption volume varies wildly between a small residential house and a large factory, K-Means clustering is utilized to group clients by their **behavioral consumption patterns** (e.g., 9-to-5 commercial vs. residential). 
* Data is aggregated to create an "average 24-hour profile" for each client.
* Min-Max scaling is applied row-wise to eliminate the "volume" effect, isolating the shape of the demand curve before clustering.

### 3. Baseline Modeling: Linear Regression (`1_linear_regression.ipynb`)
A Linear Regression model is established as a baseline to determine the predictive power of standard autoregressive features.
* **Feature Creation:** Lagged variables (`Lag_15min`, `Lag_24h`) and rolling averages (`Rolling_Mean_4h`) are generated to capture historical consumption patterns. Categorical variables are then One-Hot Encoded.
* **Chronological Split:** The dataset is strictly split chronologically (Train < 2014, Test >= 2014) to prevent data leakage.
* **Recursive Forecasting:** To simulate real-world conditions, predictions are made step-by-step. The model's own predictions are fed back into the lag and rolling mean features for future steps, preventing the model from "cheating" by looking at actual future values.

### 4. Advanced Modeling: Prophet (`2_Prophet.ipynb`)
Meta's Prophet library is utilized for robust time-series forecasting.
* Built-in country holiday effects are applied (`m.add_country_holidays(country_name='PT')`).
* Custom regressors are integrated, including the engineered `Temp_National_Avg` and the autoregressive features (`Lag_15min`, `Lag_24h`, `Rolling_Mean_4h`).
* Similar to the baseline, evaluation is conducted using strict recursive forecasting on a holdout set, and performance is evaluated using Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## 🚀 Running the Repo

### 1. Create a virtual environment

If you are using VS Code, open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`), select **Python: Create Environment**, and choose **Venv**. VS Code will create a `.venv` folder and activate it automatically in all new terminals.

Alternatively, from the terminal:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order

Open the `notebooks/` folder and execute them sequentially:

| Step | Notebook | Description |
|------|----------|-------------|
| 0 | `0_data_preprocessing.ipynb` | Processes raw data, engineers features, fetches weather, outputs `processed_electricity_data.parquet` |
| 0.5 | `0.5_clustering.ipynb` | Segments clients by behavior and volume, outputs `client_clusters.csv` and `client_size_categories.csv` |
| 1 | `1_linear_regression.ipynb` | Trains and evaluates the autoregressive Linear Regression baseline |
| 2 | `2_Prophet.ipynb` | Trains and evaluates the Meta Prophet time-series model |

---
**Main libraries used:**
* Data manipulation & computation: `pandas`, `numpy`, `pyarrow`
* Visualization: `matplotlib`, `seaborn`
* Machine Learning & Forecasting: `scikit-learn`, `prophet`
* Utilities: `python-dateutil`, `requests`, `tqdm`