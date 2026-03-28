"""
split_and_cluster.py
--------------------
Stage 2 of the data pipeline.

Reads the fully-processed parquet produced by process_data.py
(which already contains Cluster and Consumer_Category columns),
applies the chronological train / test split, and writes two
partitioned files ready for any modelling script.

Output
------
Datasets/train_final.parquet   — rows with Date.year <  2014
Datasets/test_final.parquet    — rows with Date.year >= 2014

Usage
-----
    python scripts/split_and_cluster.py

    # Override default paths
    python scripts/split_and_cluster.py \\
        --input  Datasets/processed_electricity_data.parquet \\
        --outdir Datasets \\
        --cutoff 2014
"""

import argparse
import os
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def split_and_save(input_path: str, output_dir: str, cutoff_year: int) -> None:
    """
    Load the processed parquet, split by year, and export train / test files.

    The split boundary follows the same convention used throughout the project:
        train  →  Date.year <  cutoff_year
        test   →  Date.year >= cutoff_year

    Parameters
    ----------
    input_path : str
        Path to the fully-processed parquet file
        (output of process_data.py).
    output_dir : str
        Directory where train_final.parquet and test_final.parquet
        will be written.
    cutoff_year : int
        First year that belongs to the test set (default: 2014).
    """
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"Loading processed data from: {input_path}")
    t0 = time.time()
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Sanity checks
    # ------------------------------------------------------------------
    required_cols = {"ClientID", "Date", "Consumption", "Cluster", "Consumer_Category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input parquet is missing expected columns: {missing}\n"
            "Did you run process_data.py first?"
        )

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    print(f"Splitting on year < {cutoff_year} (train) / >= {cutoff_year} (test)...")
    train = df[df["Date"].dt.year <  cutoff_year].copy()
    test  = df[df["Date"].dt.year >= cutoff_year].copy()

    n_clients_train = train["ClientID"].nunique()
    n_clients_test  = test["ClientID"].nunique()
    print(f"  Train: {len(train):>12,} rows | {n_clients_train} clients "
          f"| {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"  Test:  {len(test):>12,} rows | {n_clients_test} clients "
          f"| {test['Date'].min().date()} → {test['Date'].max().date()}")

    # ------------------------------------------------------------------
    # 4. Cluster distribution summary (quick sanity check)
    # ------------------------------------------------------------------
    print("\n  Cluster distribution (train):")
    cluster_summary = (
        train.groupby("Cluster", observed=True)["ClientID"]
        .nunique()
        .rename("n_clients")
        .reset_index()
    )
    for _, row in cluster_summary.iterrows():
        print(f"    Cluster {int(row['Cluster'])}: {int(row['n_clients'])} clients")

    print("\n  Consumer_Category distribution (train):")
    cat_summary = (
        train.groupby("Consumer_Category", observed=True)["ClientID"]
        .nunique()
        .rename("n_clients")
        .reset_index()
    )
    for _, row in cat_summary.iterrows():
        print(f"    {row['Consumer_Category']}: {int(row['n_clients'])} clients")

    # ------------------------------------------------------------------
    # 5. Export
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_final.parquet")
    test_path  = os.path.join(output_dir, "test_final.parquet")

    print(f"\nExporting train → {train_path}")
    train.to_parquet(train_path, index=False)

    print(f"Exporting test  → {test_path}")
    test.to_parquet(test_path, index=False)

    total = time.time() - t0
    print(f"\n✓ Done in {total:.1f}s")
    print(f"  train_final.parquet : {os.path.getsize(train_path) / 1e6:.1f} MB")
    print(f"  test_final.parquet  : {os.path.getsize(test_path)  / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the processed electricity parquet into train / test files."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "Datasets", "processed_electricity_data.parquet"),
        help="Path to the processed parquet (default: Datasets/processed_electricity_data.parquet)",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(PROJECT_ROOT, "Datasets"),
        help="Output directory for train_final.parquet and test_final.parquet (default: Datasets/)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=2014,
        help="First year assigned to the test set (default: 2014)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_and_save(
        input_path=args.input,
        output_dir=args.outdir,
        cutoff_year=args.cutoff,
    )
