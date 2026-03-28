"""
visualization.py
----------------
Shared plotting utilities for all forecasting models.

All functions accept pre-computed evaluation DataFrames (typically the output
of an evaluate_models() step) and produce matplotlib figures inline or to disk.
Keeping visualizations here avoids duplicating plot logic across LR, Prophet,
SARIMAX, and future models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cluster_portfolio(
    cluster_eval: pd.DataFrame,
    summary: pd.DataFrame,
    model_label: str = "Prediction",
    n_steps_to_show: int = 1344,
) -> None:
    """
    Plot actual vs predicted portfolio load for each cluster.

    Shows the last `n_steps_to_show` time-steps (default = 1344, i.e. ~14 days
    at 15-min resolution) to keep charts readable.

    Parameters
    ----------
    cluster_eval : pd.DataFrame
        One row per (Cluster, Date) with columns ['Actual_kW', 'Predicted_kW'].
    summary : pd.DataFrame
        Indexed by Cluster with columns ['Portfolio_MAPE', 'Portfolio_WMAPE'].
    model_label : str, optional
        Label used in the plot legend to identify the model (e.g. "LR", "Prophet").
    n_steps_to_show : int, optional
        Number of trailing time-steps to plot per cluster for readability.
    """
    unique_clusters = sorted(cluster_eval["Cluster"].unique())
    fig, axes = plt.subplots(len(unique_clusters), 1, figsize=(15, 5 * len(unique_clusters)))

    if len(unique_clusters) == 1:
        axes = [axes]

    for idx, cluster_id in enumerate(unique_clusters):
        ax = axes[idx]
        c_plot = cluster_eval[cluster_eval["Cluster"] == cluster_id].sort_values("Date")
        c_mape  = summary.loc[cluster_id, "Portfolio_MAPE"]
        c_wmape = summary.loc[cluster_id, "Portfolio_WMAPE"]
        plot_slice = -n_steps_to_show

        ax.plot(
            c_plot["Date"].values[plot_slice:],
            c_plot["Actual_kW"].values[plot_slice:],
            label="Actual Portfolio Load",
            color="steelblue",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            c_plot["Date"].values[plot_slice:],
            c_plot["Predicted_kW"].values[plot_slice:],
            label=f"{model_label} (Day-Ahead)",
            color="tomato",
            linestyle="--",
            alpha=0.9,
            linewidth=1.5,
        )

        ax.set_title(
            f"Cluster {cluster_id} Portfolio — MAPE: {c_mape:.2f}% | WMAPE: {c_wmape:.2f}%",
            fontsize=14,
        )
        ax.set_ylabel("Total Consumption (kW)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()


def analyze_time_periods(
    test: pd.DataFrame,
    n_bins: int = 4,
) -> None:
    """
    Break the test set into equal-length time periods and visualise the spread
    of Absolute Percentage Error (APE) and Absolute Error (kW) across periods.

    Parameters
    ----------
    test : pd.DataFrame
        Test DataFrame with columns ['Date', 'Actual_kW', 'Predicted_kW'].
        Rows where either column is NaN are dropped automatically.
    n_bins : int, optional
        Number of time periods to split the test window into. Default is 4.
    """
    df_eval = test.dropna(subset=["Actual_kW", "Predicted_kW"]).copy()
    df_eval = df_eval.sort_values("Date")

    _, bin_edges = pd.cut(df_eval["Date"], bins=n_bins, retbins=True)
    bin_edges = pd.to_datetime(bin_edges)
    dynamic_labels = [
        f"{bin_edges[i].strftime('%b %d')} to {bin_edges[i+1].strftime('%b %d')}"
        for i in range(n_bins)
    ]
    df_eval["Time_Period"] = pd.cut(df_eval["Date"], bins=n_bins, labels=dynamic_labels)

    portfolio_ts = (
        df_eval.groupby(["Date", "Time_Period"], observed=True)[["Actual_kW", "Predicted_kW"]]
        .sum()
        .reset_index()
    )
    portfolio_ts["Abs_Error"] = np.abs(portfolio_ts["Actual_kW"] - portfolio_ts["Predicted_kW"])
    portfolio_ts["APE"] = (portfolio_ts["Abs_Error"] / portfolio_ts["Actual_kW"]) * 100

    print("\n--- PORTFOLIO PERFORMANCE BY TIME PERIOD (Aggregated) ---")
    portfolio_wmape = portfolio_ts.groupby("Time_Period", observed=True).apply(
        lambda x: (x["Abs_Error"].sum() / x["Actual_kW"].sum()) * 100
    )
    for period, wmape_val in portfolio_wmape.items():
        avg_mape = portfolio_ts[portfolio_ts["Time_Period"] == period]["APE"].mean()
        print(f"  {period}:   WMAPE = {wmape_val:.2f}% | MAPE = {avg_mape:.2f}%")

    df_eval["Abs_Error"] = np.abs(df_eval["Actual_kW"] - df_eval["Predicted_kW"])
    mask_ape = df_eval["Actual_kW"] > 0.1
    df_ape = df_eval[mask_ape].copy()
    df_ape["APE"] = (df_ape["Abs_Error"] / df_ape["Actual_kW"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df_ape, x="Time_Period", y="APE", ax=axes[0], showfliers=False)
    axes[0].set_title("Individual Client Percentage Error Spread (APE)", fontsize=14)
    axes[0].set_ylabel("APE (%)")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=df_eval, x="Time_Period", y="Abs_Error", ax=axes[1], showfliers=False)
    axes[1].set_title("Individual Client Volume Error Spread (kW)", fontsize=14)
    axes[1].set_ylabel("Absolute Error (kW)")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()
