"""
src/rwa_sa.py

Implements the Basel III Standardised Approach (SA) risk weight
function for residential real estate exposures.

Regulatory source:
    BCBS, Basel III: Finalising post-crisis reforms, December 2017
    Table 12: Risk weights for residential real estate exposures
    (regulatory retail, not reliant on cash flows from property)

The SA assigns risk weights based solely on LTV at origination.
There is no adjustment for borrower credit quality (FICO).
This FICO-blindness is the core motivation for the output floor:
a high-FICO borrower and a low-FICO borrower with identical LTVs
receive identical SA capital treatment.

SA RWA = EAD × risk weight
where EAD = orig_upb (original unpaid principal balance)

Usage:
    from src.rwa_sa import compute_sa_rwa
    df = compute_sa_rwa(df)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import SA_LTV_BREAKPOINTS, SA_RISK_WEIGHTS


def get_sa_risk_weight(ltv: float) -> float:
    """
    Return the Basel III SA risk weight for a residential mortgage
    given its original LTV ratio (expressed as a percentage, e.g. 75.0).

    The LTV breakpoints and risk weights are defined in config.py and
    sourced from BCBS December 2017, Table 12.

    Parameters
    ----------
    ltv : float
        Original LTV ratio as a percentage (e.g. 75.0 means 75%).

    Returns
    -------
    float
        Risk weight as a decimal (e.g. 0.30 means 30%).

    Examples
    --------
    >>> get_sa_risk_weight(45.0)   # LTV <= 50% → 20%
    0.20
    >>> get_sa_risk_weight(75.0)   # LTV <= 80% → 30%
    0.30
    >>> get_sa_risk_weight(105.0)  # LTV > 100% → 70%
    0.70
    """
    if np.isnan(ltv):
        return np.nan

    # Walk through breakpoints in order — return the first weight
    # whose breakpoint the LTV does not exceed
    # SA_LTV_BREAKPOINTS = [0.50, 0.60, 0.80, 0.90, 1.00, inf]
    # but our LTV is stored as a percentage (e.g. 75), not a decimal
    # So we compare against breakpoints × 100
    for breakpoint, weight in zip(SA_LTV_BREAKPOINTS, SA_RISK_WEIGHTS):
        if ltv <= breakpoint * 100:
            return weight

    # Should never reach here given float("inf") as last breakpoint
    return SA_RISK_WEIGHTS[-1]


def compute_sa_rwa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the SA risk weight function to every loan in the dataset
    and compute SA RWA.

    Adds two columns to the DataFrame:
        sa_rw  : SA risk weight as a decimal (e.g. 0.30)
        sa_rwa : SA RWA in USD (EAD × risk weight)

    EAD proxy: orig_upb (original unpaid principal balance).
    We use origination balance rather than current balance because
    the performance file monthly balances were not retained in the
    modelling dataset. This is a standard simplification in
    academic IRB research and is noted as a limitation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: oltv (LTV %), orig_upb (USD balance)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with sa_rw and sa_rwa columns added.
    """
    df = df.copy()

    # Vectorised application — faster than row-by-row apply()
    # np.vectorize wraps our scalar function to work on arrays
    vectorised_rw = np.vectorize(get_sa_risk_weight)
    df["sa_rw"] = vectorised_rw(df["oltv"].values)

    # SA RWA = EAD × risk weight
    # orig_upb is in USD, sa_rw is a decimal
    df["sa_rwa"] = df["orig_upb"] * df["sa_rw"]

    return df


def print_sa_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of SA risk weights and RWA by FICO band.
    This is a key diagnostic — SA should show little variation
    across FICO bands since it is LTV-driven, not FICO-driven.
    That flat pattern is the setup for the output floor finding.
    """
    print("\nSA RWA summary by FICO band:")
    print(f"  {'Band':<25} {'Loans':>8} {'Mean SA RW':>12} {'Mean SA RWA':>14} {'Total SA RWA':>16}")
    print("  " + "-" * 78)

    summary = (
        df.groupby("fico_band", observed=True)
        .agg(
            loans=("sa_rw", "count"),
            mean_rw=("sa_rw", "mean"),
            mean_rwa=("sa_rwa", "mean"),
            total_rwa=("sa_rwa", "sum"),
        )
        .reset_index()
    )

    for _, row in summary.iterrows():
        print(
            f"  {str(row['fico_band']):<25} "
            f"{int(row['loans']):>8,} "
            f"{row['mean_rw']*100:>11.1f}% "
            f"${row['mean_rwa']:>13,.0f} "
            f"${row['total_rwa']:>15,.0f}"
        )

    print(f"\n  Total SA RWA across all loans: ${df['sa_rwa'].sum():,.0f}")
    print(f"  Implied capital requirement (×8%): ${df['sa_rwa'].sum() * 0.08:,.0f}")


if __name__ == "__main__":
    from config import DATA_PROCESSED

    print("=" * 60)
    print("Week 5: Standardised Approach RWA")
    print("Source: BCBS December 2017, Table 12")
    print("=" * 60)

    print("\nLoading model dataset ...")
    df = pd.read_parquet(DATA_PROCESSED / "model_dataset.parquet")
    print(f"  Loaded {len(df):,} loans")

    print("\nComputing SA risk weights and RWA ...")
    df = compute_sa_rwa(df)

    # Quick validation — check that all risk weights are valid
    valid_weights = set([w for w in SA_RISK_WEIGHTS])
    actual_weights = set(df["sa_rw"].dropna().unique())
    unexpected = actual_weights - valid_weights
    if unexpected:
        print(f"  WARNING: unexpected risk weights found: {unexpected}")
    else:
        print(f"  All risk weights valid: {sorted(actual_weights)}")

    # Distribution of LTV bands
    print("\nLTV distribution and SA risk weight mapping:")
    print(f"  {'LTV range':<20} {'Loans':>8} {'SA RW':>8} {'% of sample':>12}")
    print("  " + "-" * 52)

    ltv_labels = ["≤50%", "≤60%", "≤80%", "≤90%", "≤100%", ">100%"]
    ltv_bins   = [0, 50, 60, 80, 90, 100, float("inf")]

    df["ltv_band"] = pd.cut(
        df["oltv"],
        bins=ltv_bins,
        labels=ltv_labels,
        right=True,
    )

    ltv_summary = df.groupby("ltv_band", observed=True).size().reset_index(name="count")
    ltv_summary["pct"] = ltv_summary["count"] / len(df) * 100

    for _, row in ltv_summary.iterrows():
        # Look up the risk weight for this band
        idx = ltv_labels.index(str(row["ltv_band"]))
        rw  = SA_RISK_WEIGHTS[idx]
        print(
            f"  {str(row['ltv_band']):<20} "
            f"{int(row['count']):>8,} "
            f"{rw*100:>7.0f}% "
            f"{row['pct']:>11.1f}%"
        )

    print_sa_summary(df)

    # Save updated dataset with SA columns
    output_path = DATA_PROCESSED / "model_dataset.parquet"
    df.drop(columns=["ltv_band"]).to_parquet(output_path, index=False)
    print(f"\nSaved updated dataset to {output_path.name}")
    print("\nWeek 5 complete. Ready for src/rwa_irb.py (week 6).")