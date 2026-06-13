from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent   # the project root
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ── Freddie Mac file naming convention ──────────────────────────────────────
# Freddie Mac names files like:
#   historical_data_2013Q1.txt  (origination)
#   historical_data_time_2013Q1.txt  (performance/monthly)
SAMPLE_YEARS = [2015, 2017, 2019]   # replace the full range
# ── Column names ─────────────────────────────────────────────────────────────
# Freddie Mac origination file columns (pipe-delimited, no header)
# Source: Freddie Mac Single Family Loan-Level Dataset User Guide
ORIG_COLS = [
    "credit_score",        # FICO at origination
    "first_payment_date",
    "first_time_homebuyer",
    "maturity_date",
    "msa",                 # metropolitan statistical area
    "mip",                 # mortgage insurance percentage
    "num_units",
    "occupancy_status",
    "cltv",                # combined LTV
    "dti",                 # debt-to-income ratio
    "orig_upb",            # original unpaid principal balance (EAD proxy)
    "oltv",                # original LTV — this is our key SA variable
    "orig_interest_rate",
    "channel",
    "prepay_penalty",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",  # JOIN KEY
    "loan_purpose",
    "orig_loan_term",
    "num_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming",
    "pre_relief_refinance",     #Pre-HARP Loan Sequence Number
    "program_indicator",
    "relief_refinance",         #HARP Indicator
    "property_valuation_method",
    "io_indicator",
    "mortgage_insurance_cancellation",     #Mortgage Insurance Cancellation Indicator
]

# Freddie Mac performance file columns
PERF_COLS = [
    "loan_sequence_number",   # JOIN KEY
    "monthly_reporting_period",
    "current_upb",
    "current_loan_delinquency_status",  # DPD bucket: 0,1,2,...,XX
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",      
# 01=prepaid, 02=third party sale, 03=foreclosure,
# 06=REO, 09=charge-off — we care about 03,06,09
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "due_date_last_paid_installment",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]

#  Default definition
# 90+ DPD: current_loan_delinquency_status >= 3 (each unit = 30 days)
# Zero Balance Codes that confirm default
DEFAULT_ZBC = {"03", "09"}
DPD_THRESHOLD = 3   # 3 × 30 = 90+ days past due

#  FICO segmentation
FICO_BINS = [0, 619, 659, 719, 759, 850]
FICO_LABELS = ["<620 (subprime)", "620–659 (near-prime)",
               "660–719 (prime)", "720–759 (prime+)", "760+ (super-prime)"]

#  IRB parameters
IRB_CORRELATION_R = 0.15   # Basel fixed correlation for retail residential mortgages
IRB_LGD_BASE      = 0.20   # Base LGD assumption — 20%, justified as conservative
IRB_PD_FLOOR      = 0.0003 # Basel minimum PD floor (3 basis points)
CONFIDENCE_LEVEL  = 0.999  # Basel 99.9th percentile (one-year VaR)
SCALING_FACTOR    = 1.06   # Basel-mandated IRB scaling factor

# ── SA risk weights 
# Basel III finalised framework, Dec 2017, Table 12 (residential real estate)
SA_LTV_BREAKPOINTS    = [0.50, 0.60, 0.80, 0.90, 1.00, float("inf")]
SA_RISK_WEIGHTS       = [0.20, 0.25, 0.30, 0.40, 0.50, 0.70]

# ── Output floor ─────────────────────────────────────────────────────────────
OUTPUT_FLOOR = 0.725    # 72.5% of SA RWA
import pandas as pd

# Performance cutoff date — read directly from data, not assumed
# This is the latest default date observed in the performance files
PERFORMANCE_CUTOFF = pd.Timestamp("2025-09-01")

# Observation window per vintage — derived from data diagnostic
# (cutoff date minus Jan 1 of origination year) / 365.25
OBSERVATION_WINDOW_YEARS = {
    2015: 10.67,
    2017: 8.67,
    2019: 6.67,
}