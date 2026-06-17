# Basel III Output Floor: Heterogeneous Capital Impacts in Residential Mortgage Lending

Banks must hold capital against mortgages they issue. Basel III gives them two ways to calculate how much: the Standardised Approach (SA), a simple lookup table based on loan-to-value ratios, or the Internal Ratings-Based approach (IRB), which uses the bank's own default prediction models.

IRB-approved banks have historically produced capital figures well below SA by building models that correctly identify low-risk borrowers. Basel responded with the output floor: **your IRB capital cannot fall below 72.5% of what SA would require**.

The floor is an aggregate rule. A bank's total IRB figure just needs to reach 72.5% of its total SA figure. But this raises a question: does a portfolio-level rule bind evenly across the risk spectrum, or does it land harder on some borrowers than others?

This project shows it lands very unevenly.

<img width="1330" height="580" alt="fig9_irb_sa_ratio_by_fico" src="https://github.com/user-attachments/assets/b949d7fb-1ead-4d3a-8faa-9486051faf4c" />


The floor barely touches subprime borrowers — their aggregate IRB capital (1.54x SA) is already well above SA because the model correctly flags them as risky. But for super-prime borrowers, IRB produces very low capital (0.34x SA, accurately reflecting their near-zero default risk), and that is exactly where the floor bites hardest. At the overall portfolio level the aggregate ratio is 0.54x — the floor binds, and the binding is concentrated almost entirely in the prime and super-prime segments.

This creates a problem. If the floor constrains capital relief on the safest loans, banks face a rational incentive to securitise those loans — removing them from the balance sheet to sidestep the constraint — while retaining the riskier ones. The aggregate floor could inadvertently concentrate risk on bank balance sheets rather than reduce it.

This may be why regulators have chosen a phased implementation through 2030 rather than immediate enforcement.

---

## Data and methodology

**Data:** Freddie Mac Single-Family Loan-Level Dataset, origination years 2015, 2017, 2019 — approximately 4.46 million loans. Not included due to licence restrictions; download from (https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).

**Method:**
- SA RWA computed from Basel III Table 12 LTV-based risk weight lookup (BCBS December 2017)
- IRB PD model: Platt-calibrated logistic regression trained on loan-level origination features (AUC 0.719, mean annual PD 0.62% after horizon adjustment). Complex models (XGBoost, Random Forest) were not included in this study, as banks are not permitted to use IRB models that cannot be explained to supervisors — the only model used is logistic regression, where every variable and constant is interpretable. Details on the logistics regression model is available on outputs folder, table0.
- IRB RWA computed using the Basel supervisory Vasicek formula (BCBS June 2006, §328 — (https://www.bis.org/publ/bcbs128.htm))
- Output floor ratio (IRB/SA) computed at the aggregate level (sum of IRB RWA / sum of SA RWA), by FICO segment and for the full portfolio — consistent with the floor's design as a portfolio-level, not loan-level, rule
- H1 (heterogeneity) tested with Kruskal-Wallis on loan-level ratios; H2 (predictability) tested with OLS regression of the aggregate ratio on FICO score
- Robustness checks across LGD assumptions of 10% (Basel regulatory floor for residential mortgages, BCBS Dec 2017 §266) through 25%

**Key results:** Aggregate IRB/SA ratio declines monotonically from 1.54x (subprime) to 0.34x (super-prime) under the base case LGD of 20%. The floor binds at the portfolio level (overall ratio 0.54x) and is concentrated in the prime+ and super-prime segments. This heterogeneity is highly predictable from FICO score distributions (band-level R² = 0.997, β = −0.015, p < 0.001).

<img width="1480" height="730" alt="fig14_robustness_ratio_by_lgd" src="https://github.com/user-attachments/assets/8fd0ec3d-ecb3-4316-8f65-20bb55f5d93e" />


**Stack:** Python — pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib

**Regulatory sources:** (https://www.bis.org/bcbs/publ/d424.htm) · (https://www.bis.org/publ/bcbs128.htm)
