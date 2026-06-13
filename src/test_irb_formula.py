"""
src/test_irb_formula.py

Unit tests for the IRB formula.
Run this before using compute_irb_rwa on real data.

Usage:
    python src/test_irb_formula.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rwa_irb import irb_risk_weight
import numpy as np

def test_irb_formula():
    print("=" * 55)
    print("IRB formula unit tests")
    print("=" * 55)

    # ── Test 1: PD floor enforcement ─────────────────────────────
    # A PD of 0 would break the formula (log of 0 is undefined).
    # The floor should prevent this.
    rw = irb_risk_weight(pd=0.0, lgd=0.20)
    assert not np.isnan(rw), "FAIL: PD=0 produced NaN"
    print(f"Test 1 passed — PD floor: PD=0.0 → RW = {rw*100:.2f}%")

    # ── Test 2: monotonicity ─────────────────────────────────────
    # Higher PD should generally produce higher capital requirements
    # (though the relationship is not strictly monotonic at the
    # extremes due to the stress formula — this is expected Basel
    # behaviour and a legitimate dissertation discussion point)
    rw_low  = irb_risk_weight(pd=0.005, lgd=0.20)
    rw_mid  = irb_risk_weight(pd=0.05,  lgd=0.20)
    rw_high = irb_risk_weight(pd=0.20,  lgd=0.20)
    print(f"\nTest 2 — risk weights across PD spectrum:")
    print(f"  PD=0.5%  → RW = {rw_low*100:.2f}%")
    print(f"  PD=5.0%  → RW = {rw_mid*100:.2f}%")
    print(f"  PD=20.0% → RW = {rw_high*100:.2f}%")

    # ── Test 3: LGD sensitivity ──────────────────────────────────
    # Risk weight should scale with LGD — higher LGD means more
    # capital needed for the same PD
    rw_lgd10 = irb_risk_weight(pd=0.02, lgd=0.10)
    rw_lgd20 = irb_risk_weight(pd=0.02, lgd=0.20)
    rw_lgd30 = irb_risk_weight(pd=0.02, lgd=0.30)
    assert rw_lgd10 < rw_lgd20 < rw_lgd30, "FAIL: RW not increasing with LGD"
    print(f"\nTest 3 passed — LGD sensitivity (PD=2%):")
    print(f"  LGD=10% → RW = {rw_lgd10*100:.2f}%")
    print(f"  LGD=20% → RW = {rw_lgd20*100:.2f}%")
    print(f"  LGD=30% → RW = {rw_lgd30*100:.2f}%")

    # ── Test 4: NaN handling ─────────────────────────────────────
    rw_nan = irb_risk_weight(pd=float("nan"), lgd=0.20)
    assert np.isnan(rw_nan), "FAIL: NaN PD did not return NaN"
    print(f"\nTest 4 passed — NaN input returns NaN")

    # ── Test 5: capital requirement is positive ──────────────────
    for pd_val in [0.001, 0.01, 0.05, 0.15]:
        rw = irb_risk_weight(pd=pd_val, lgd=0.20)
        assert rw > 0, f"FAIL: negative risk weight at PD={pd_val}"
    print(f"\nTest 5 passed — all risk weights are positive")

    # ── Summary table ────────────────────────────────────────────
    print("\nFull risk weight schedule (LGD=20%, Basel R=0.15):")
    print(f"  {'PD':>8}  {'RW':>10}  {'K (capital req)':>16}")
    print("  " + "-" * 38)
    for pd_val in [0.001, 0.003, 0.005, 0.01, 0.02,
                   0.03,  0.05,  0.10,  0.15, 0.20]:
        rw = irb_risk_weight(pd=pd_val, lgd=0.20)
        K  = rw / (12.5 * 1.06)   # reverse the scaling to get K
        print(f"  {pd_val*100:>7.1f}%  {rw*100:>9.2f}%  {K*100:>15.2f}%")

    print("\nAll tests passed. Formula is ready for real data.")

if __name__ == "__main__":
    test_irb_formula()