"""
src/rwa_irb.py

Implements the Basel III Internal Ratings-Based (IRB) supervisory
risk weight formula for retail residential mortgage exposures.

Regulatory source:
    BCBS, Basel II: International Convergence of Capital Measurement
    and Capital Standards, June 2006, paragraphs 328-330.
    (The retail IRB formula was not changed in the December 2017
    finalisation — only the SA was revised.)

Economic foundation:
    The formula is an implementation of the Vasicek (2002) single-factor
    credit model. Every borrower's asset value is driven by:
        (1) A systemic factor — the state of the economy (shared by all)
        (2) An idiosyncratic factor — borrower-specific randomness

    Default occurs when asset value falls below a threshold.
    Capital must cover losses in the 99.9th percentile stress scenario.

Fixed parameters for retail residential mortgages (set by Basel, not estimated):
    R    = 0.15   asset correlation
    M    = 1      effective maturity (retail exposures use 1 by convention)
    sf   = 1.06   Basel scaling factor (introduced in Basel II 2004)

Usage:
    from src.rwa_irb import compute_irb_rwa
    df = compute_irb_rwa(df, pd_column="pd_logistic", lgd=0.20)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats   # we need norm.cdf and norm.ppf

from config import (
    IRB_CORRELATION_R,
    IRB_LGD_BASE,
    IRB_PD_FLOOR,
    CONFIDENCE_LEVEL,
    SCALING_FACTOR,
    DATA_PROCESSED,
)


# ── Core formula ─────────────────────────────────────────────────────────────

def irb_risk_weight(pd: float, lgd: float) -> float:
    """
    Compute the Basel IRB risk weight for a single retail residential
    mortgage exposure.

    The full formula (BCBS June 2006, paragraph 328):

        K  = LGD × N( sqrt(R/(1-R)) × G(PD) + sqrt(1/(1-R)) × G(0.999) )
             − PD × LGD

        RW = K × 12.5 × 1.06

    where:
        N()   = standard normal CDF    (scipy: norm.cdf)
        G()   = inverse normal CDF     (scipy: norm.ppf)
        R     = 0.15 (asset correlation, fixed by Basel for retail residential)
        0.999 = 99.9th percentile confidence level (Basel stress scenario)
        12.5  = 1 / 0.08 (converts capital ratio to risk weight)
        1.06  = Basel scaling factor

    Parameters
    ----------
    pd  : float  Probability of default (e.g. 0.01 = 1%). Must be > 0.
    lgd : float  Loss given default as decimal (e.g. 0.20 = 20%).

    Returns
    -------
    float  Risk weight as a decimal (e.g. 0.35 = 35% risk weight).
           Returns NaN if pd is NaN.
    """

    # ── Guard: missing PD ────────────────────────────────────────────────────
    if np.isnan(pd):
        return np.nan

    # ── Step 1: enforce the Basel PD floor ───────────────────────────────────
    # Basel paragraph 285 states PD cannot be below 0.03% (3 basis points).
    # Without this floor, very low PD loans would produce near-zero capital
    # requirements, which regulators consider imprudent.
    # Our IRB_PD_FLOOR = 0.0003 from config.py.
    pd = max(pd, IRB_PD_FLOOR)

    # ── Step 2: compute the two scaling terms ────────────────────────────────
    # These come directly from the Vasicek model derivation.
    #
    # R = 0.15: asset correlation. 15% of variance in borrower asset value
    # is explained by the common systemic (economy-wide) factor.
    # The remaining 85% is idiosyncratic (borrower-specific).
    R = IRB_CORRELATION_R   # 0.15

   # term_a multiplies G(PD) — scales the borrower's PD contribution
    # sqrt(1 / (1-R)) = sqrt(1 / 0.85) = 1.0847
    term_a = np.sqrt(1 / (1 - R))

    # term_b multiplies G(0.999) — scales the systemic stress contribution
    # sqrt(R / (1-R)) = sqrt(0.15 / 0.85) = 0.4201
    term_b = np.sqrt(R / (1 - R))

    # ── Step 3: convert PD to a z-score ─────────────────────────────────────
    # G(PD) = norm.ppf(PD) is the z-score below which PD% of a standard
    # normal distribution lies. For low PD this is a large negative number.
    #
    # Example:  PD = 1%  → G(0.01) = -2.326
    #           PD = 5%  → G(0.05) = -1.645
    #           PD = 50% → G(0.50) =  0.000
    #
    # The negative sign reflects that safe borrowers have asset values
    # well above the default threshold under normal conditions.
    g_pd = stats.norm.ppf(pd)

    # ── Step 4: compute the Basel stress threshold ───────────────────────────
    # G(0.999) = the z-score at the 99.9th percentile of the systemic factor.
    # This represents a severe recession — a year so bad it occurs only
    # once in 1000 years. Basel requires capital to survive this scenario.
    # G(0.999) ≈ 3.0902 (fixed constant — does not change loan to loan)
    g_999 = stats.norm.ppf(CONFIDENCE_LEVEL)   # CONFIDENCE_LEVEL = 0.999

    # ── Step 5: compute the conditional default threshold ────────────────────
    # Under the 99.9th percentile stress scenario, the conditional default
    # threshold for this borrower is:
    #
    #   inner = term_a × G(PD) + term_b × G(0.999)
    #
    # This is the z-score such that N(inner) gives the conditional PD
    # under the stress scenario — i.e. the probability this borrower
    # defaults given that the economy is in its worst 0.1% state.
# inner = sqrt(1/(1-R)) × G(PD) + sqrt(R/(1-R)) × G(0.999)
    inner = term_a * g_pd + term_b * g_999

    # ── Step 6: compute the conditional PD under stress ──────────────────────
    # N(inner) = the conditional probability of default under the stress
    # scenario. This is always higher than the unconditional PD.
    #
    # For a borrower with PD = 1%:
    #   inner ≈ 0.4201 × (-2.326) + 1.0847 × 3.090 = -0.977 + 3.352 = 2.375
    #   N(2.375) ≈ 0.991 → 99.1% conditional default rate under stress
    conditional_pd = stats.norm.cdf(inner)

    # ── Step 7: compute the capital requirement K ────────────────────────────
    # K = Expected Loss under stress − Expected Loss under normal conditions
    #   = LGD × conditional_PD − LGD × unconditional_PD
    #   = LGD × (conditional_PD − PD)
    #
    # This is the UNEXPECTED loss — the part that capital must absorb.
    # Expected losses should be covered by loan pricing and provisioning,
    # not capital. Capital covers the difference between stressed and
    # expected outcomes.
    K = lgd * conditional_pd - pd * lgd

    # ── Step 8: convert capital requirement to a risk weight ─────────────────
    # The Basel minimum capital ratio is 8% (Pillar 1).
    # RWA is defined such that: Capital = RWA × 8%
    # Therefore: RWA = Capital / 0.08 = Capital × 12.5
    #
    # The 1.06 scaling factor was introduced in Basel II (2004) to ensure
    # IRB approaches generate sufficient capital in aggregate. It is a
    # regulatory calibration, not derived from the model.
    risk_weight = K * 12.5 * SCALING_FACTOR   # SCALING_FACTOR = 1.06

    return risk_weight


# ── Vectorised application ────────────────────────────────────────────────────

def compute_irb_rwa(df: pd.DataFrame,
                    pd_column: str,
                    lgd: float = IRB_LGD_BASE,
                    rw_col: str = None,
                    rwa_col: str = None) -> pd.DataFrame:
    """
    Apply the IRB risk weight formula to every loan in the dataset
    and compute IRB RWA.

    Parameters
    ----------
    df        : pd.DataFrame  Must contain pd_column and orig_upb.
    pd_column : str           Name of the column containing PD estimates.
                              e.g. "pd_logistic", "pd_xgb", "pd_rf"
    lgd       : float         Loss given default assumption (decimal).
                              Defaults to IRB_LGD_BASE = 0.20 from config.
    rw_col    : str           Output column name for risk weight.
                              Defaults to "irb_rw_{pd_column}".
    rwa_col   : str           Output column name for RWA.
                              Defaults to "irb_rwa_{pd_column}".

    Returns
    -------
    pd.DataFrame with two new columns added: irb_rw and irb_rwa.
    """

    df = df.copy()

    # Set default output column names if not provided
    if rw_col is None:
        rw_col  = f"irb_rw_{pd_column}"
    if rwa_col is None:
        rwa_col = f"irb_rwa_{pd_column}"

    # Vectorise our scalar function so it works on the whole column at once
    # This is equivalent to looping but runs in compiled C — much faster
    vectorised_irb = np.vectorize(irb_risk_weight)

    df[rw_col]  = vectorised_irb(df[pd_column].values, lgd)
    df[rwa_col] = df["orig_upb"] * df[rw_col]

    return df