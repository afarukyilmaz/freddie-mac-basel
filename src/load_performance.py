"""

Reads the Freddie Mac monthly performance files in chunks,
identifies whether each loan ever defaulted (Basel III definition:
90+ DPD or Zero Balance Code 03/06/09), and writes a compact
one-row-per-loan summary to data/processed/performance_summary.parquet.

This script handles files too large to load into RAM by processing
one chunk at a time and aggregating incrementally.

Usage:
    python src/load_performance.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

# Import our central config — never hardcode paths
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    SAMPLE_YEARS,
    PERF_COLS,
    DEFAULT_ZBC,
    DPD_THRESHOLD,
)


# ── Helper: parse delinquency status ────────────────────────────────────────

def parse_dpd(value) -> int:
    """
    Convert the delinquency status string to an integer.

    Known values in this Freddie Mac vintage (confirmed by diagnostic):
      "0"–"67" → numeric DPD buckets (each unit = 30 days)
      "RA"     → REO Acquisition — property seized, definitely defaulted
      NaN      → missing, treat as non-default

    Returns -1 for unparseable values.
    """
    if pd.isna(value):
        return -1

    cleaned = str(value).strip().upper()

    if cleaned == "RA":
        return 24    # well above DPD_THRESHOLD of 3 — flags as default

    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return -1

# ── Helper: classify default in one chunk ───────────────────────────────────

def classify_defaults_in_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Given a chunk of the performance file (many rows per loan),
    return a DataFrame with one row per loan and two columns:
      - ever_defaulted   : 1 if the loan ever hit the default definition
      - first_default_date : the earliest month it defaulted (NaT if never)

    The Basel III retail default definition (CRR Art. 178) requires
    90+ days past due OR a credit loss event (ZBC 03/06/09).
    """

    # --- Flag: delinquency-based default ---
    # Apply our parser to every row, then flag rows >= threshold (3 = 90 DPD)
    chunk["dpd_int"] = chunk["current_loan_delinquency_status"].apply(parse_dpd)
    chunk["dpd_default"] = chunk["dpd_int"] >= DPD_THRESHOLD

    # --- Flag: zero balance code default ---
    # ZBC is stored as a string; strip whitespace to be safe
    chunk["zbc_clean"] = chunk["zero_balance_code"].astype(str).str.strip()
    chunk["zbc_default"] = chunk["zbc_clean"].isin(DEFAULT_ZBC)

    # --- Combined default flag ---
    chunk["is_default_event"] = chunk["dpd_default"] | chunk["zbc_default"]

    # --- Parse the reporting period as a date ---
    # Format is YYYYMM as an integer e.g. 201503 = March 2015
    chunk["report_date"] = pd.to_datetime(
        chunk["monthly_reporting_period"].astype(str),
        format="%Y%m",
        errors="coerce",
    )

    # --- Aggregate to one row per loan ---
    # For each loan, we want:
    #   ever_defaulted = 1 if any row had a default event
    #   first_default_date = the earliest report_date where is_default_event = True

    # Separate defaulted rows for the date calculation
    defaulted_rows = chunk[chunk["is_default_event"]]

    # Aggregate ever_defaulted flag
    default_flag = (
        chunk.groupby("loan_sequence_number")["is_default_event"]
        .max()                          # max of True/False = True if any row was True
        .astype(int)
        .rename("ever_defaulted")
    )

    # Aggregate first default date (only for loans that defaulted)
    if len(defaulted_rows) > 0:
        first_default = (
            defaulted_rows.groupby("loan_sequence_number")["report_date"]
            .min()
            .rename("first_default_date")
        )
    else:
        # No defaults in this chunk — create an empty Series with correct dtype
        first_default = pd.Series(
            dtype="datetime64[ns]", name="first_default_date"
        )

    # Join the two aggregations
    result = default_flag.to_frame().join(first_default, how="left")

    return result


# ── Main loader ──────────────────────────────────────────────────────────────

def load_performance_files(chunksize: int = 500_000) -> pd.DataFrame:
    """
    Iterate over all quarterly performance files for SAMPLE_YEARS,
    read each in chunks, classify defaults, and accumulate results.

    Parameters
    ----------
    chunksize : int
        Number of rows to read per chunk. 500,000 rows uses roughly
        500 MB RAM. Reduce to 250_000 if you run into memory errors.

    Returns
    -------
    pd.DataFrame
        One row per unique loan_sequence_number with columns:
        ever_defaulted (int), first_default_date (datetime or NaT).
    """

    # We will accumulate one summary DataFrame per chunk
    # and concatenate at the end — more memory-efficient than
    # building one giant DataFrame
    summaries = []

    quarters = ["Q1", "Q2", "Q3", "Q4"]

    for year in SAMPLE_YEARS:
        for q in quarters:

            perf_file = DATA_RAW / f"historical_data_time_{year}{q}.txt"

            if not perf_file.exists():
                print(f"  Skipping {perf_file.name} — not found")
                continue

            print(f"\nProcessing {perf_file.name} ...")
            file_size_gb = perf_file.stat().st_size / 1e9
            print(f"  File size: {file_size_gb:.2f} GB")

            # pd.read_csv with chunksize returns an iterator
            # We never load the whole file — only chunksize rows at a time
            chunk_iter = pd.read_csv(
                perf_file,
                sep="|",                    # Freddie Mac uses pipe delimiter
                header=None,                # No header row in raw files
                names=PERF_COLS,
                dtype=str,                  # Read everything as string first
                                            # avoids silent type coercion bugs
                chunksize=chunksize,
            )

            # tqdm wraps the iterator to show a progress bar
            # We don't know total chunks upfront so no total= arg
            for chunk in tqdm(chunk_iter, desc=f"  {year}{q}"):

                chunk_summary = classify_defaults_in_chunk(chunk)
                summaries.append(chunk_summary)

                # Explicitly delete the chunk and run garbage collection
                # Python's garbage collector handles most memory, but with
                # DataFrames this large, being explicit helps
                del chunk
                gc.collect()

    print("\nConcatenating all chunk summaries ...")

    # Stack all chunk summaries and resolve conflicts:
    # The same loan can appear across multiple chunks (different months).
    # After concat we have many rows per loan — we need to re-aggregate.
    combined = pd.concat(summaries, ignore_index=False)

    print("Re-aggregating across chunks ...")

    # Group again at the full-dataset level
    final_flag = (
        combined.groupby("loan_sequence_number")["ever_defaulted"]
        .max()
    )

    final_date = (
        combined.groupby("loan_sequence_number")["first_default_date"]
        .min()
    )

    performance_summary = final_flag.to_frame().join(final_date, how="left")

    return performance_summary


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Week 2: Performance file loader")
    print("Basel III default definition: 90+ DPD or ZBC 03/06/09")
    print("=" * 60)

    # Run the loader
    summary = load_performance_files(chunksize=500_000)

    # Quick sanity checks before saving
    total_loans    = len(summary)
    total_defaults = summary["ever_defaulted"].sum()
    default_rate   = total_defaults / total_loans * 100

    print(f"\nSanity check:")
    print(f"  Total unique loans : {total_loans:,}")
    print(f"  Total defaults     : {total_defaults:,}")
    print(f"  Overall default rate: {default_rate:.2f}%")

    # Expected: default rate between 1% and 6% for this period
    # If it's 0% something is wrong with the default flags
    # If it's >15% something is wrong with the DPD parsing
    if default_rate < 0.5:
        print("\n  WARNING: default rate seems very low — check parse_dpd()")
    elif default_rate > 15:
        print("\n  WARNING: default rate seems very high — check DPD_THRESHOLD")
    else:
        print("\n  Default rate looks plausible. Proceeding to save.")

    # Save as Parquet — preserves dtypes, much faster than CSV to reload
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "performance_summary.parquet"
    summary.to_parquet(output_path)

    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
    print("\nWeek 2 complete. Ready for src/build_dataset.py (week 3).")


if __name__ == "__main__":
    main()