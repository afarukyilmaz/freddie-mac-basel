import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_RAW, SAMPLE_YEARS, PERF_COLS

def check_unique_dpd_values():
    for year in SAMPLE_YEARS:
        for q in ["Q1"]:
            perf_file = DATA_RAW / f"historical_data_time_{year}{q}.txt"
            if not perf_file.exists():
                continue

            print(f"Scanning {perf_file.name} ...")
            print("Reading first 2 million rows\n")

            df = pd.read_csv(
                perf_file,
                sep="|",
                header=None,
                names=PERF_COLS,
                dtype=str,
                nrows=2_000_000,
            )

            total_rows = len(df)
            print(f"Total rows read: {total_rows:,}\n")

            # ── Delinquency status frequencies ──────────────────────────
            print("Delinquency status value counts (sorted by frequency):")
            print(f"  {'Value':<10} {'Count':>10} {'% of rows':>10}")
            print("  " + "-" * 34)

            dpd_counts = (
                df["current_loan_delinquency_status"]
                .fillna("NaN")
                .value_counts()
                .sort_values(ascending=False)
            )

            for val, count in dpd_counts.items():
                pct = count / total_rows * 100
                print(f"  {str(val):<10} {count:>10,} {pct:>9.2f}%")

            # ── Zero balance code frequencies ────────────────────────────
            print("\nZero balance code value counts (sorted by frequency):")
            print(f"  {'Value':<10} {'Count':>10} {'% of rows':>10}")
            print("  " + "-" * 34)

            zbc_counts = (
                df["zero_balance_code"]
                .fillna("NaN/blank")
                .value_counts()
                .sort_values(ascending=False)
            )

            for val, count in zbc_counts.items():
                pct = count / total_rows * 100
                print(f"  {str(val):<10} {count:>10,} {pct:>9.2f}%")

            print("\n--- Interpretation notes ---")
            print("Delinquency '0'  = current (expect this to dominate)")
            print("Delinquency '3+' = 90+ DPD → default trigger")
            print("Delinquency 'RA' = REO Acquisition → default trigger")
            print("ZBC 'NaN/blank'  = loan still active (most rows)")
            print("ZBC '03'         = Short Sale/Charge Off → default")
            print("ZBC '09'         = REO Disposition → default")

            return

if __name__ == "__main__":
    check_unique_dpd_values()