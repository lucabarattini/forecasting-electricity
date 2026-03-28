"""
evaluation.py
-------------
Shared evaluation metrics for all forecasting models in the project.

All functions operate on raw (un-scaled) kW values and return percentages.
Keeping metrics here ensures a single source of truth across Linear Regression,
Prophet, SARIMAX, and any future model.
"""
import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Rows where y_true <= threshold are excluded to avoid division-by-zero
    distortions on near-zero consumption readings.

    Parameters
    ----------
    y_true : array-like
        Actual consumption values (kW).
    y_pred : array-like
        Predicted consumption values (kW).
    threshold : float, optional
        Minimum actual value to include in the calculation. Default is 0.1 kW.

    Returns
    -------
    float
        MAPE expressed as a percentage (e.g. 5.3 means 5.3%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true > threshold
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE).

    Weights each observation by its actual volume, which makes this metric
    more robust to low-consumption periods and preferred for portfolio-level
    business reporting.

        WMAPE = ( sum|actual - pred| / sum|actual| ) * 100

    Parameters
    ----------
    y_true : array-like
        Actual consumption values (kW).
    y_pred : array-like
        Predicted consumption values (kW).

    Returns
    -------
    float
        WMAPE expressed as a percentage (e.g. 3.7 means 3.7%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return np.nan

    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def compute_cluster_metrics(cluster_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAPE and WMAPE for each cluster from a pre-aggregated evaluation
    DataFrame (one row per cluster-date with Actual_kW and Predicted_kW columns).

    This is a convenience wrapper around mape() and wmape() that operates at
    cluster level, suitable for summary tables in notebooks or reports.

    Parameters
    ----------
    cluster_eval : pd.DataFrame
        DataFrame with columns: ['Cluster', 'Date', 'Actual_kW', 'Predicted_kW'].
        Typically the output of the evaluate_models() step.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame indexed by Cluster with columns:
        ['Portfolio_MAPE', 'Portfolio_WMAPE'].
    """
    records = []
    for cluster_id, group in cluster_eval.groupby("Cluster", observed=True):
        cluster_mape  = mape(group["Actual_kW"].values, group["Predicted_kW"].values)
        cluster_wmape = wmape(group["Actual_kW"].values, group["Predicted_kW"].values)
        records.append({
            "Cluster": cluster_id,
            "Portfolio_MAPE":  round(cluster_mape,  2),
            "Portfolio_WMAPE": round(cluster_wmape, 2),
        })

    summary = pd.DataFrame(records).set_index("Cluster")
    return summary


def print_global_metrics(cluster_eval: pd.DataFrame) -> None:
    """
    Print global (all-cluster, all-dates) MAPE and WMAPE to stdout.

    Parameters
    ----------
    cluster_eval : pd.DataFrame
        DataFrame with columns: ['Actual_kW', 'Predicted_kW', 'Abs_Error', 'Perc_Error'].
    """
    global_wmape_val = wmape(cluster_eval["Actual_kW"].values, cluster_eval["Predicted_kW"].values)
    global_mape_val  = cluster_eval["Perc_Error"].mean()
    print(f"\nGlobal Portfolio MAPE:  {global_mape_val:.2f}%")
    print(f"Global Portfolio WMAPE: {global_wmape_val:.2f}%")
