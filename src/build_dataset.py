"""
src/build_dataset.py

Joins the Freddie Mac origination files to the performance summary
produced in week 2, cleans key variables, creates FICO segments,
and writes the final modelling dataset to data/processed/.

Outputs:
    data/processed/model_dataset.parquet   — full cleaned dataset
    data/processed/train.parquet           — 80% stratified train split
    data/processed/test.parquet            — 20% stratified test split

Usage:
    python src/build_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_RAW,
    DATA_PROCESSED,
    SAMPLE_YEARS,
    ORIG_COLS,
    FICO_BINS,
    FICO_LABELS,
)


# ── Step 1: load all origination files ──────────────────────────────────────

def load_origination_files() -> pd.DataFrame:
    """
    Read all quarterly origination files for SAMPLE_YEARS and
    concatenate into one DataFrame.

    The origination file has one row per loan — no chunking needed
    since it is much smaller than the performance file.
    """
    frames = []
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    for year in SAMPLE_YEARS:
        for q in quarters:
            orig_file = DATA_RAW / f"historical_data_{year}{q}.txt"

            if not orig_file.exists():
                print(f"  Skipping {orig_file.name} — not found")
                continue

            print(f"  Loading {orig_file.name} ...")

            df = pd.read_csv(
                orig_file,
                sep="|",
                header=None,
                names=ORIG_COLS,
                dtype=str,       # read as string first, clean types below
                low_memory=False,
            )

            # Tag each loan with its origination year — useful for
            # vintage analysis in the robustness checks later
            df["orig_year"] = year

            frames.append(df)

    origination = pd.concat(frames, ignore_index=True)
    print(f"\n  Total origination rows loaded: {len(origination):,}")
    return origination


# ── Step 2: clean and type-cast key columns ──────────────────────────────────

def clean_origination(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the four columns we use in the analysis:
      - credit_score  (FICO)
      - oltv          (original LTV — used for SA risk weight)
      - dti           (debt-to-income ratio — IRB model feature)
      - orig_upb      (original unpaid principal balance — EAD proxy)

    Freddie Mac encodes missing values as 9999 (credit_score, oltv, dti)
    or as blank strings. We convert these to NaN and then drop rows
    where any of the four key columns are missing, since we cannot
    compute SA RWA or run IRB models without them.
    """

    df = df.copy()

    # ── credit_score (FICO) ──────────────────────────────────────────────
    df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")
    # 9999 is Freddie Mac's missing value code
    df["credit_score"] = df["credit_score"].replace(9999, np.nan)
    # FICO valid range is 300–850; flag anything outside as missing
    df.loc[~df["credit_score"].between(300, 850), "credit_score"] = np.nan

    # ── oltv (original LTV) ──────────────────────────────────────────────
    df["oltv"] = pd.to_numeric(df["oltv"], errors="coerce")
    df["oltv"] = df["oltv"].replace(999, np.nan)
    # LTV must be positive; above 200 is almost certainly a data error
    df.loc[~df["oltv"].between(1, 200), "oltv"] = np.nan

    # ── dti (debt-to-income) ─────────────────────────────────────────────
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    df["dti"] = df["dti"].replace(999, np.nan)
    # DTI above 100 is not meaningful (debt cannot exceed income by this
    # measure in normal underwriting)
    df.loc[~df["dti"].between(1, 100), "dti"] = np.nan

    # ── orig_upb (original balance — our EAD proxy) ──────────────────────
    df["orig_upb"] = pd.to_numeric(df["orig_upb"], errors="coerce")
    df.loc[df["orig_upb"] <= 0, "orig_upb"] = np.nan

    # ── orig_interest_rate ───────────────────────────────────────────────
    df["orig_interest_rate"] = pd.to_numeric(
        df["orig_interest_rate"], errors="coerce"
    )

    # ── Drop rows missing any key column ────────────────────────────────
    key_cols = ["credit_score", "oltv", "dti", "orig_upb", "loan_sequence_number"]
    before = len(df)
    df = df.dropna(subset=key_cols)
    after = len(df)
    dropped = before - after
    print(f"  Dropped {dropped:,} rows ({dropped/before*100:.1f}%) missing key columns")

    return df


# ── Step 3: create FICO band variable ────────────────────────────────────────

def add_fico_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment loans into FICO bands using the cutoffs defined in config.py.
    This is the key grouping variable for the output floor analysis.

    Bands (industry standard, justified in dissertation methodology):
      <620        subprime
      620–659     near-prime
      660–719     prime
      720–759     prime+
      760+        super-prime
    """
    df = df.copy()

    df["fico_band"] = pd.cut(
        df["credit_score"],
        bins=FICO_BINS,
        labels=FICO_LABELS,
        right=True,        # intervals are (left, right] — so 620 goes into
                           # the 620-659 band, not the <620 band
        include_lowest=True,
    )

    return df


# ── Step 4: join origination to performance summary ──────────────────────────

def join_performance(orig: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the origination data onto the performance summary.

    We use a left join because:
    - Every row in orig is a loan we want to keep
    - Some loans may not appear in performance_summary if their
      performance files were not downloaded — these get NaN for
      ever_defaulted and we drop them
    - Loans that appear in origination but have no performance record
      are unusable for modelling
    """
    perf_path = DATA_PROCESSED / "performance_summary.parquet"
    print(f"\n  Loading performance summary from {perf_path.name} ...")
    perf = pd.read_parquet(perf_path)
    print(f"  Performance summary shape: {perf.shape}")

    # Reset index so loan_sequence_number becomes a regular column
    perf = perf.reset_index()

    merged = orig.merge(
        perf,
        on="loan_sequence_number",
        how="left",
    )

    # Loans with no performance record are unusable
    before = len(merged)
    merged = merged.dropna(subset=["ever_defaulted"])
    after = len(merged)
    print(f"  Dropped {before - after:,} loans with no performance record")

    # Convert ever_defaulted to integer (was float after merge)
    merged["ever_defaulted"] = merged["ever_defaulted"].astype(int)

    return merged


# ── Step 5: train/test split ─────────────────────────────────────────────────

def split_dataset(df: pd.DataFrame):
    """
    Split into 80% train, 20% test.

    We stratify on both fico_band AND ever_defaulted jointly.
    This ensures each FICO segment is represented proportionally
    in both splits, and that the rare default class is not
    accidentally concentrated in one split.
    """

    # Create a joint stratification key
    # e.g. "760+ (super-prime)_0" or "< 620 (subprime)_1"
    df = df.copy()
    df["strat_key"] = (
        df["fico_band"].astype(str) + "_" + df["ever_defaulted"].astype(str)
    )

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,     # fixed seed — results are reproducible
        stratify=df["strat_key"],
    )

    # Drop the helper column — it was only needed for stratification
    train = train.drop(columns=["strat_key"])
    test  = test.drop(columns=["strat_key"])

    return train, test


# ── Step 6: sanity checks ────────────────────────────────────────────────────

def print_sanity_checks(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame):
    """
    Print a summary table so we can verify the dataset looks correct
    before saving. These numbers go into your dissertation methodology.
    """
    print("\n" + "=" * 55)
    print("Dataset summary")
    print("=" * 55)
    print(f"  Total loans          : {len(df):,}")
    print(f"  Train set            : {len(train):,} ({len(train)/len(df)*100:.0f}%)")
    print(f"  Test set             : {len(test):,}  ({len(test)/len(df)*100:.0f}%)")
    print(f"  Overall default rate : {df['ever_defaulted'].mean()*100:.2f}%")
    print(f"  Train default rate   : {train['ever_defaulted'].mean()*100:.2f}%")
    print(f"  Test default rate    : {test['ever_defaulted'].mean()*100:.2f}%")

    print("\n  Default rate by FICO band:")
    print(f"  {'Band':<25} {'Loans':>8} {'Defaults':>9} {'Rate':>7}")
    print("  " + "-" * 52)

    band_stats = (
        df.groupby("fico_band", observed=True)["ever_defaulted"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "loans", "sum": "defaults", "mean": "rate"})
    )

    for band, row in band_stats.iterrows():
        print(
            f"  {str(band):<25} {int(row['loans']):>8,} "
            f"{int(row['defaults']):>9,} {row['rate']*100:>6.2f}%"
        )

    print("\n  Key feature statistics:")
    for col in ["credit_score", "oltv", "dti", "orig_upb"]:
        print(f"  {col:<20} mean={df[col].mean():>8.1f}  "
              f"std={df[col].std():>7.1f}  "
              f"missing={df[col].isna().sum():>6,}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Week 3: Build modelling dataset")
    print("=" * 55)

    print("\n[1/5] Loading origination files ...")
    orig = load_origination_files()

    print("\n[2/5] Cleaning key columns ...")
    orig = clean_origination(orig)

    print("\n[3/5] Adding FICO bands ...")
    orig = add_fico_band(orig)

    print("\n[4/5] Joining performance summary ...")
    dataset = join_performance(orig)

    print("\n[5/5] Splitting into train/test ...")
    train, test = split_dataset(dataset)

    print_sanity_checks(dataset, train, test)

    # ── Save outputs ─────────────────────────────────────────────────────
    print("\nSaving files ...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(DATA_PROCESSED / "model_dataset.parquet", index=False)
    train.to_parquet(DATA_PROCESSED / "train.parquet",           index=False)
    test.to_parquet(DATA_PROCESSED / "test.parquet",             index=False)

    for name in ["model_dataset", "train", "test"]:
        path = DATA_PROCESSED / f"{name}.parquet"
        print(f"  {name}.parquet — {path.stat().st_size / 1e6:.1f} MB")

    print("\nWeek 3 complete. Ready for notebooks/01_eda.ipynb (week 4).")


if __name__ == "__main__":
    main()