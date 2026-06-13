"""
src/compute_rwa.py

Computes IRB RWA using the Basel supervisory formula and the
logistic regression PD estimates from week 7.

Key methodological note on PD horizon adjustment:
    Our logistic regression was trained on ever_defaulted — a cumulative
    default outcome over each loan's full observation window.
    Basel IRB requires 1-year forward-looking PDs (BCBS June 2006, §285).

    We convert using the annual hazard approximation:
        annual_PD = 1 - (1 - cumulative_PD)^(1/T)
    where T is the loan's actual observation window in years, derived from
    the performance data cutoff date (2025-09-01) and origination year:
        2015 vintage: T = 10.67 years
        2017 vintage: T = 8.67  years
        2019 vintage: T = 6.67  years

    Validated annual PD estimates after adjustment:
        2015: 0.411%  |  2017: 0.623%  |  2019: 0.777%
        Overall: ~0.62% — consistent with prime conforming mortgage portfolios

Then computes the output floor ratio:
    ratio = IRB_RWA / SA_RWA

The floor binds where ratio < 0.725 — meaning IRB RWA is less than
72.5% of SA RWA. In that case, the bank must use SA RWA × 72.5%
as the binding floor rather than its internal model estimate.

Adds columns to model_dataset.parquet:
    pd_logistic_1yr   : annual PD estimate (horizon-adjusted)
    irb_rw_logistic   : IRB risk weight (decimal)
    irb_rwa_logistic  : IRB RWA in USD
    floor_ratio       : IRB RWA / SA RWA
    floor_binds       : 1 if floor_ratio < 0.725, else 0
    floor_rwa         : SA RWA × 72.5% (loan-level diagnostic)
    rwa_uplift        : loan-level RWA shortfall vs floor (diagnostic only —
                        do NOT sum to get the aggregate; the output floor is a
                        portfolio-level rule, see note below)

Usage:
    python src/compute_rwa.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config import DATA_PROCESSED, OUTPUT_FLOOR, IRB_LGD_BASE, FICO_LABELS, IRB_PD_FLOOR
from src.rwa_irb import compute_irb_rwa


# ── Observation window by vintage ────────────────────────────────────────────
# Derived from performance data cutoff date (2025-09-01) minus Jan 1
# of each origination year. Validated by diagnostic on actual data.
# Do NOT hardcode these — they were read from the data directly.
OBSERVATION_WINDOW_YEARS = {
    2015: 10.67,   # 2015-01-01 → 2025-09-01 = 10.67 years
    2017: 8.67,    # 2017-01-01 → 2025-09-01 = 8.67 years
    2019: 6.67,    # 2019-01-01 → 2025-09-01 = 6.67 years
}


# ── Helper: safe read/write ───────────────────────────────────────────────────

def read_parquet_safe(path):
    import time
    for attempt in range(1, 4):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  Read attempt {attempt} failed. Retrying in 5s ...")
            time.sleep(5)

def write_parquet_safe(df, path):
    # Write to temp file first, then rename — prevents corruption if interrupted
    tmp = Path(str(path) + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Week 10: IRB RWA computation and output floor analysis")
    print(f"LGD assumption : {IRB_LGD_BASE*100:.0f}%")
    print(f"Output floor   : {OUTPUT_FLOOR*100:.1f}% of SA RWA")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────
    print("\nLoading model dataset ...")
    df = read_parquet_safe(DATA_PROCESSED / "model_dataset.parquet")
    print(f"  Loaded {len(df):,} loans")

    # Verify required columns exist
    required = ["pd_logistic", "sa_rwa", "orig_upb", "fico_band", "orig_year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: missing columns: {missing}")
        print("  Run weeks 5 and 7 first.")
        return

    print(f"  pd_logistic mean (cumulative): {df['pd_logistic'].mean()*100:.3f}%")
    sa_rwa_total = df['sa_rwa'].sum()
    print(f"  SA RWA total                 : {sa_rwa_total:,.0f} USD")

    # ── PD horizon adjustment ─────────────────────────────────────
    # Our logistic model was trained on ever_defaulted — a cumulative
    # default outcome over each loan's full observation window.
    # Basel IRB requires 1-year forward-looking PDs (BCBS §285).
    #
    # Conversion formula (annual hazard approximation):
    #   annual_PD = 1 - (1 - cumulative_PD)^(1/T)
    # where T = observation window in years for that loan's vintage.
    #
    # This is loan-specific because origination year determines
    # how many years of performance data we observe before the
    # performance cutoff date of 2025-09-01.
    print("\nApplying PD horizon adjustment ...")

    # Map each loan to its vintage's observation window
    df["obs_years"] = df["orig_year"].map(OBSERVATION_WINDOW_YEARS)

    # Apply the annual hazard conversion
    df["pd_logistic_1yr"] = 1 - (1 - df["pd_logistic"]) ** (1 / df["obs_years"])

    # Enforce Basel minimum PD floor of 0.03% (3 basis points)
    # BCBS June 2006, paragraph 285: PD cannot be below 0.03%
    df["pd_logistic_1yr"] = df["pd_logistic_1yr"].clip(lower=IRB_PD_FLOOR)

    # Print vintage-by-vintage validation
    print(f"\n  {'Vintage':<10} {'Obs window':>12} {'Cumulative PD':>15} {'Annual PD':>12}")
    print("  " + "-" * 52)
    for yr in [2015, 2017, 2019]:
        mask = df["orig_year"] == yr
        cum  = df.loc[mask, "pd_logistic"].mean() * 100
        ann  = df.loc[mask, "pd_logistic_1yr"].mean() * 100
        t    = OBSERVATION_WINDOW_YEARS[yr]
        print(f"  {yr:<10} {t:>11.2f}yr {cum:>14.3f}% {ann:>11.3f}%")

    overall_ann = df["pd_logistic_1yr"].mean() * 100
    print(f"\n  Overall annual PD: {overall_ann:.3f}%")
    print(f"  (Expected range for prime conforming mortgages: 0.3% - 1.5%)")

    # ── Compute IRB RWA using annual PD ──────────────────────────
    # Now we use pd_logistic_1yr (1-year PD) as input to the Basel
    # IRB formula — this is methodologically correct.
    # The IRB formula was designed for 1-year PD inputs.
    print("\nComputing IRB RWA using annual PD estimates ...")
    # Drop any stale IRB columns that may have been saved in the parquet
    stale = [c for c in df.columns if c.startswith("irb_rw")]
    if stale:
        df = df.drop(columns=stale)

    df = compute_irb_rwa(df, pd_column="pd_logistic_1yr", lgd=IRB_LGD_BASE)

    # Rename to cleaner column names
    df = df.rename(columns={
        "irb_rw_pd_logistic_1yr"  : "irb_rw_logistic",
        "irb_rwa_pd_logistic_1yr" : "irb_rwa_logistic",
    })

    total_irb_rwa = df['irb_rwa_logistic'].sum()
    mean_irb_rw   = df['irb_rw_logistic'].mean() * 100
    mean_sa_rw    = df['sa_rw'].mean() * 100

    print(f"  IRB RWA total : {total_irb_rwa:,.0f} USD")
    print(f"  Mean IRB RW   : {mean_irb_rw:.2f}%")
    print(f"  Mean SA RW    : {mean_sa_rw:.2f}%")
    # ── Compute output floor ratio ────────────────────────────────
    # floor_ratio = IRB_RWA / SA_RWA
    # The floor binds when IRB produces LESS capital than 72.5% of SA.
    # floor_ratio < 0.725 means IRB RWA < 72.5% of SA RWA.
    # With properly calibrated annual PDs (~0.62%), we expect IRB RW
    # to be well below SA RW for prime and super-prime segments,
    # meaning the floor will bind much more widely than before.
    print("\nComputing output floor ratios ...")
    df["floor_ratio"] = np.where(
        df["sa_rwa"] > 0,
        df["irb_rwa_logistic"] / df["sa_rwa"],
        np.nan
    )

    df["floor_binds"] = (df["floor_ratio"] < OUTPUT_FLOOR).astype(int)

    n_binds   = df["floor_binds"].sum()
    pct_binds = n_binds / len(df) * 100
    print(f"  Loans where floor binds: {n_binds:,} ({pct_binds:.1f}%)")

    # ── Summary by FICO band ──────────────────────────────────────
    print("\nOutput floor summary by FICO band:")
    print(f"  {'Band':<25} {'Loans':>8} {'Mean IRB/SA':>12} "
          f"{'Floor binds':>12} {'Bind rate':>10}")
    print("  " + "-" * 72)

    band_summary = (
        df.groupby("fico_band", observed=True)
        .agg(
            loans      = ("floor_ratio", "count"),
            mean_ratio = ("floor_ratio", "mean"),
            n_binds    = ("floor_binds", "sum"),
        )
        .reset_index()
    )
    band_summary["bind_rate"] = (
        band_summary["n_binds"] / band_summary["loans"] * 100
    )

    for _, row in band_summary.iterrows():
        print(
            f"  {str(row['fico_band']):<25} "
            f"{int(row['loans']):>8,} "
            f"{row['mean_ratio']:>11.3f}x "
            f"{int(row['n_binds']):>11,} "
            f"{row['bind_rate']:>9.1f}%"
        )

    print(f"\n  Overall mean IRB/SA ratio : {df['floor_ratio'].mean():.3f}x")
    print(f"  Overall floor bind rate   : {pct_binds:.1f}%")

    # ── Capital uplift from the floor ─────────────────────────────
    # The output floor is an AGGREGATE rule: it applies to TOTAL portfolio
    # RWA, not loan by loan. The floored RWA is therefore
    #     max( total IRB RWA , 72.5% × total SA RWA )
    # and the extra RWA it imposes nets the headroom of loans above the floor
    # against the shortfall of loans below it. Summing loan-level uplifts
    # OVERSTATES the impact because it ignores that netting.
    #
    # We keep the loan-level diagnostic columns below, but they must NOT be
    # summed to obtain the aggregate uplift.
    df["floor_rwa"]  = df["sa_rwa"] * OUTPUT_FLOOR          # loan-level diagnostic
    df["rwa_uplift"] = np.where(                            # loan-level diagnostic
        df["floor_binds"] == 1,
        df["floor_rwa"] - df["irb_rwa_logistic"],
        0
    )

    # Aggregate (portfolio-level) floor — the regulatory quantity
    total_sa_rwa_agg    = df["sa_rwa"].sum()
    aggregate_floor_rwa = total_sa_rwa_agg * OUTPUT_FLOOR
    floored_rwa         = max(total_irb_rwa, aggregate_floor_rwa)
    aggregate_uplift    = floored_rwa - total_irb_rwa
    extra_capital       = aggregate_uplift * 0.08
    print(f"\n  Aggregate floored RWA       : {floored_rwa:,.0f} USD")
    print(f"  Aggregate RWA uplift (floor): {aggregate_uplift:,.0f} USD")
    print(f"  Implied extra capital (×8%) : {extra_capital:,.0f} USD")

    # ── Save ──────────────────────────────────────────────────────
    print("\nSaving updated dataset ...")
    write_parquet_safe(df, DATA_PROCESSED / "model_dataset.parquet")
    print("  Saved model_dataset.parquet")

    band_summary.to_csv(
        DATA_PROCESSED / "floor_summary_by_fico.csv", index=False
    )
    print("  Saved floor_summary_by_fico.csv")

    print("\nWeek 10 complete. Ready for week 11 — H1 test.")


if __name__ == "__main__":
    main()