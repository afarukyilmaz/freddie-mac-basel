"""
src/validate_data.py

Run this after downloading Freddie Mac files to confirm all
expected origination and performance files are present.

Usage:
    python src/validate_data.py
"""

import sys
from config import DATA_RAW, SAMPLE_YEARS


def check_files() -> None:
    """
    Verify origination and performance files exist for all
    quarters across SAMPLE_YEARS.
    """
    all_present = True
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    print(f"\n{'File':<45}  {'Status'}")
    print("-" * 55)

    for year in SAMPLE_YEARS:
        for q in quarters:
            orig_file = DATA_RAW / f"historical_data_{year}{q}.txt"
            perf_file = DATA_RAW / f"historical_data_time_{year}{q}.txt"

            for f in [orig_file, perf_file]:
                status = "✓ found" if f.exists() else "✗ MISSING"
                if not f.exists():
                    all_present = False
                print(f"{f.name:<45}  {status}")

    print()
    if all_present:
        print("All files present. Ready for week 2.")
    else:
        print("Some files missing — check downloads and file naming.")
        sys.exit(1)

if __name__ == "__main__":
    check_files()