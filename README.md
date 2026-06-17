# Basel III Output Floor: Heterogeneous Capital Impacts in Residential Mortgage Lending

**MSc Finance Dissertation — University of Greenwich, 2026**

## Research Question

Does the Basel III 72.5% output floor bind heterogeneously across mortgage 
risk segments defined by FICO score, and is this heterogeneity predictable 
from observable credit score distributions?

## Hypotheses

**H1:** The output floor binds heterogeneously across FICO segments — 
supported with strong statistical evidence (Kruskal-Wallis p < 0.001).

**H2:** This heterogeneity is predictable from FICO score distributions — 
supported (loan-level OLS: β = -0.015, p < 0.001, R² = 0.674; 
band-level R² = 0.997).

## Key Finding

The IRB/SA RWA ratio declines monotonically from 4.1x (subprime) to 1.4x 
(super-prime). The output floor binds exclusively among super-prime borrowers 
under the base case LGD assumption, reflecting the fundamental asymmetry 
between the FICO-blind SA and the PD-sensitive IRB approach. This result is 
robust across all LGD assumptions tested (10%–25%).

## Data

Freddie Mac Single-Family Loan-Level Dataset, origination years 2015, 2017, 
2019. Approximately 4.46 million loan observations.

**Data not included** due to licence restrictions. Download from:  
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

## Methodology

| Step | Description | Script |
|------|-------------|--------|
| 1 | Performance file loading, default flagging | `src/load_performance.py` |
| 2 | Dataset construction, FICO segmentation | `src/build_dataset.py` |
| 3 | Standardised Approach RWA (Basel III Table 12) | `src/rwa_sa.py` |
| 4 | IRB risk weight formula (Vasicek model) | `src/rwa_irb.py` |
| 5 | Logistic regression PD model (Platt-calibrated) | `src/models/logistic_model.py` |
| 6 | IRB RWA computation and floor ratio | `src/compute_rwa.py` |
| 7 | H1 analysis — floor heterogeneity | `notebooks/04_h1_floor_heterogeneity.ipynb` |
| 8 | H2 analysis — FICO predictability | `notebooks/05_h2_predictability.ipynb` |
| 9 | Robustness checks — LGD sensitivity | `notebooks/06_robustness.ipynb` |

## Setup

```bash
git clone https://github.com/afarukyilmaz/freddie-mac-basel.git
cd freddie-mac-basel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

Run scripts in order after downloading data:

```bash
python src/load_performance.py    # Week 2: ~5 min
python src/build_dataset.py       # Week 3: ~3 min
python src/rwa_sa.py              # Week 5: ~1 min
python src/test_irb_formula.py    # Week 6: validation
python src/models/logistic_model.py  # Week 7: ~5 min
python src/compute_rwa.py         # Week 10: ~1 min
# Then run notebooks 04, 05, 06 in Jupyter
```

## Results Summary

| FICO Segment | Mean IRB/SA Ratio | Floor Bind Rate |
|---|---|---|
| <620 (subprime) | 4.12x | 0.0% |
| 620–659 (near-prime) | 3.57x | 0.0% |
| 660–719 (prime) | 2.71x | 0.0% |
| 720–759 (prime+) | 1.98x | 0.0% |
| 760+ (super-prime) | 1.40x | 1.5% |

## Regulatory Context

The Basel III output floor (BCBS December 2017) requires that IRB RWA 
cannot fall below 72.5% of SA RWA. Implementation: 2025 (UK), phased 
to 2030 (EU). This paper provides empirical evidence on the 
distributional capital impact of this constraint.

## Technical Stack

Python 3.14 · pandas · numpy · scikit-learn · scipy · statsmodels · 
matplotlib · seaborn · pyarrow

## Author

Ahmet Faruk Yilmaz — MSc Finance, University of Greenwich  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
