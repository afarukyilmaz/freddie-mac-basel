"""
src/models/logistic_model.py

Logistic regression PD model — Week 7.

This is the baseline IRB PD model. Logistic regression is chosen as
the first model for three reasons:
    1. Interpretability: coefficients have direct economic meaning
    2. Regulatory preference: Basel IRB documentation requires models
       whose outputs can be audited and explained to supervisors
    3. Baseline: provides a benchmark for XGBoost and Random Forest

The model predicts ever_defaulted (0/1) from origination features.
Output probabilities are used as PD estimates in the IRB formula.

Usage:
    python src/models/logistic_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve

from config import DATA_PROCESSED, OUTPUTS_DIR

# ── Feature set ───────────────────────────────────────────────────────────────
# These are the origination features available at the time of loan decision.
# We only use information known at origination — not post-origination data.
# This is critical: using future information would cause data leakage.
#
# Expected signs:
#   credit_score      : negative  (higher FICO → lower PD)
#   oltv              : positive  (higher LTV → higher PD)
#   dti               : positive  (higher DTI → higher PD)
#   orig_upb          : ambiguous (larger loans may be riskier or safer)
#   orig_interest_rate: positive  (higher rate → borrower was deemed riskier)
FEATURES = [
    "credit_score",
    "oltv",
    "dti",
    "orig_upb",
    "orig_interest_rate",
]

TARGET = "ever_defaulted"


# ── Helper: safe parquet reader with retry ────────────────────────────────────

def read_parquet_safe(path, retries: int = 3, wait: int = 5) -> pd.DataFrame:
    """
    Read a parquet file with automatic retry on timeout.
    On Mac, large parquet files occasionally hit a file system timeout
    (errno 60) if the file was recently written. Retrying after a short
    wait resolves this reliably.

    Parameters
    ----------
    path    : Path to parquet file
    retries : number of attempts before raising
    wait    : seconds to wait between attempts
    """
    for attempt in range(1, retries + 1):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  Read attempt {attempt} failed ({e}). "
                  f"Retrying in {wait}s ...")
            time.sleep(wait)


# ── Helper: safe parquet writer using temp file ───────────────────────────────

def write_parquet_safe(df: pd.DataFrame, path) -> None:
    """
    Write a parquet file safely using a temp file + rename pattern.
    This prevents corrupting the original file if the write is interrupted.

    Steps:
        1. Write to a temp file (.tmp) in the same directory
        2. On success, rename temp file to the target path
        3. If writing fails, the original file is untouched
    """
    tmp_path = Path(str(path) + ".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(path)


# ── Step 1: load train and test sets ─────────────────────────────────────────

def load_data():
    """
    Load the train and test parquet files created in week 3.
    Drop any rows where features are missing — logistic regression
    cannot handle NaN values.
    """
    train = read_parquet_safe(DATA_PROCESSED / "train.parquet")
    test  = read_parquet_safe(DATA_PROCESSED / "test.parquet")

    before_train = len(train)
    train = train.dropna(subset=FEATURES + [TARGET])
    test  = test.dropna(subset=FEATURES + [TARGET])

    dropped = before_train - len(train)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with missing feature values")

    print(f"  Train: {len(train):,} loans  |  "
          f"Default rate: {train[TARGET].mean()*100:.2f}%")
    print(f"  Test:  {len(test):,} loans   |  "
          f"Default rate: {test[TARGET].mean()*100:.2f}%")

    return train, test


# ── Step 2: build and train the pipeline ─────────────────────────────────────

def build_and_train(train: pd.DataFrame) -> Pipeline:
    """
    Build a sklearn Pipeline with two steps:
        1. StandardScaler     — normalises features to zero mean, unit variance
        2. LogisticRegression — fits the log-odds model

    Why StandardScaler?
    Logistic regression uses gradient-based optimisation. If features are on
    very different scales (FICO: 300-850, orig_upb: $50,000-$600,000), the
    optimiser struggles. Scaling makes optimisation faster and more stable.
    Note: scaling does not change predictions, only training efficiency.

    Why class_weight='balanced'?
    Default rate is ~4.83%. Without correction, the model learns to predict
    'never default' and achieves 95% accuracy but zero recall on defaults.
    'balanced' reweights each class inversely proportional to its frequency:
        weight_default     = n_samples / (2 × n_defaults)
        weight_non_default = n_samples / (2 × n_non_defaults)
    This forces the model to treat both classes proportionally.
    NOTE: this inflates raw predicted probabilities — we fix this with
    Platt scaling calibration in step 6.

    Why max_iter=1000?
    The default is 100 iterations. With 3.5 million training loans the
    solver needs more iterations to converge.

    Why solver='lbfgs'?
    L-BFGS is a quasi-Newton optimisation method — well-suited for
    medium-large datasets with a moderate number of features.
    """
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
                       class_weight = "balanced",
                       max_iter     = 1000,
                       solver       = "lbfgs",
                       random_state = 42,
                   )),
    ])

    X_train = train[FEATURES]
    y_train = train[TARGET]

    print("  Training logistic regression ...")
    model_pipeline.fit(X_train, y_train)
    print("  Training complete.")

    return model_pipeline


# ── Step 3: evaluate on test set ─────────────────────────────────────────────

def evaluate(model_pipeline: Pipeline, test: pd.DataFrame):
    """
    Evaluate the trained model on the held-out test set.

    Metrics:
        AUC-ROC : Area under ROC curve. Measures discrimination.
                  Random = 0.5, perfect = 1.0. Target: > 0.70.
        Gini    : 2 × AUC − 1. Standard credit risk metric.
        KS stat : Maximum separation between default/non-default distributions.
        Brier   : Mean squared error of probability predictions.
        Log-loss: Cross-entropy loss — penalises confident wrong predictions.
    """
    X_test = test[FEATURES]
    y_test = test[TARGET]

    # predict_proba returns [P(class=0), P(class=1)] — we take column 1
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks = (tpr - fpr).max()

    brier = brier_score_loss(y_test, y_prob)
    ll    = log_loss(y_test, y_prob)

    metrics = {"auc": auc, "gini": gini, "ks": ks,
               "brier": brier, "logloss": ll}

    print(f"\n  Model performance on test set (pre-calibration):")
    print(f"  {'AUC-ROC':<12}: {auc:.4f}")
    print(f"  {'Gini':<12}: {gini:.4f}")
    print(f"  {'KS stat':<12}: {ks:.4f}")
    print(f"  {'Brier score':<12}: {brier:.4f}")
    print(f"  {'Log-loss':<12}: {ll:.4f}")

    return metrics, y_prob


# ── Step 4: inspect coefficients ─────────────────────────────────────────────

def print_coefficients(model_pipeline: Pipeline) -> None:
    """
    Print the logistic regression coefficients.

    The coefficient on feature X = change in log-odds of default
    for a one-unit increase in X (after scaling).

    Expected signs:
        credit_score       : NEGATIVE (higher FICO → lower default risk)
        oltv               : POSITIVE (higher LTV → higher default risk)
        dti                : POSITIVE (higher DTI → higher default risk)
        orig_upb           : to be determined by data
        orig_interest_rate : POSITIVE (higher rate signals riskier loan)
    """
    model  = model_pipeline.named_steps["model"]
    coeffs = model.coef_[0]

    print(f"\n  Logistic regression coefficients (log-odds scale):")
    print(f"  {'Feature':<22} {'Coefficient':>12}  {'Direction'}")
    print("  " + "-" * 50)

    for feature, coef in zip(FEATURES, coeffs):
        direction = "↓ lower PD" if coef < 0 else "↑ higher PD"
        print(f"  {feature:<22} {coef:>12.4f}  {direction}")

    print(f"\n  Intercept: {model.intercept_[0]:.4f}")


# ── Step 5: plots ─────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, auc: float) -> None:
    """Plot the ROC curve and save to outputs/."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2c7bb6", linewidth=2,
            label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--",
            linewidth=1, label="Random classifier (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — logistic regression PD model")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = OUTPUTS_DIR / "fig6_roc_logistic.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


def plot_calibration(y_test, y_prob, suffix: str = "",
                     title_suffix: str = "") -> None:
    """
    Plot the calibration curve (reliability diagram).
    A well-calibrated model follows the diagonal: when it predicts
    5% PD, approximately 5% of those loans actually default.
    """
    prob_true, prob_pred = calibration_curve(
        y_test, y_prob, n_bins=10, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, color="#2c7bb6",
            linewidth=2, marker="o", markersize=5,
            label=f"Logistic regression{title_suffix}")
    ax.plot([0, prob_pred.max()], [0, prob_pred.max()],
            color="#aaaaaa", linestyle="--", linewidth=1,
            label="Perfect calibration")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Fraction actually defaulted")
    ax.set_title(f"Calibration plot — logistic regression{title_suffix}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = OUTPUTS_DIR / f"fig7_calibration_logistic{suffix}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


def plot_pd_distribution(y_prob, y_test, suffix: str = "") -> None:
    """
    Plot the distribution of predicted PDs for defaulted vs
    non-defaulted loans. A good model shows clear separation.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(y_prob[y_test == 0], bins=60, alpha=0.6, color="#2c7bb6",
            label="Non-defaulted", density=True)
    ax.hist(y_prob[y_test == 1], bins=60, alpha=0.6, color="#d7191c",
            label="Defaulted",     density=True)

    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of predicted PDs — logistic regression")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = OUTPUTS_DIR / f"fig8_pd_distribution_logistic{suffix}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path.name}")


# ── Step 6: calibrate and generate PD estimates ───────────────────────────────

def calibrate_and_generate_pd(model_pipeline: Pipeline,
                               train: pd.DataFrame,
                               test: pd.DataFrame) -> None:
    """
    Apply Platt scaling to correct probability inflation from
    class_weight='balanced', then generate calibrated PD estimates
    for the full dataset.

    The problem:
        class_weight='balanced' improves discrimination (AUC) but shifts
        raw probabilities far above the true base rate (~4.83%).
        The uncalibrated model predicts ~43% mean PD — about 9x too high.

    The fix — Platt scaling:
        Fit a second logistic regression that maps raw model scores to
        calibrated probabilities matching the true base rate.
        This preserves ranking (AUC unchanged) while correcting the scale.

    Basel relevance:
        IRB PD estimates must reflect actual long-run default rates.
        Uncalibrated PDs would be rejected by any model validation team.
    """
    print("\n  Calibrating model with Platt scaling ...")

    # Step 1: get raw uncalibrated probabilities on the TRAIN set
    # We use train (not test) to avoid data leakage into calibration
    raw_train_probs = model_pipeline.predict_proba(train[FEATURES])[:, 1]

    # Step 2: fit a logistic calibration layer
    # This single-feature logistic regression maps raw scores → calibrated PDs
    # It learns the correct intercept and slope to match the true base rate
    calibration_lr = LogisticRegression(max_iter=1000)
    calibration_lr.fit(raw_train_probs.reshape(-1, 1), train[TARGET])

    # Step 3: define the two-stage prediction function
    # Stage 1: original pipeline predicts raw score
    # Stage 2: calibration layer maps raw score → calibrated PD
    def calibrated_predict_proba(X):
        raw = model_pipeline.predict_proba(X)[:, 1]
        return calibration_lr.predict_proba(raw.reshape(-1, 1))[:, 1]

    # ── Evaluate calibrated model on test set ────────────────────
    y_test = test[TARGET].values
    y_prob_cal = calibrated_predict_proba(test[FEATURES])

    auc_cal = roc_auc_score(y_test, y_prob_cal)
    print(f"  Calibrated AUC             : {auc_cal:.4f}  "
          f"(should match pre-calibration AUC)")
    print(f"  Calibrated mean PD on test : {y_prob_cal.mean()*100:.3f}%")
    print(f"  True default rate          : {y_test.mean()*100:.3f}%")

    # Generate updated calibration plot
    plot_calibration(y_test, y_prob_cal,
                     suffix="_calibrated",
                     title_suffix=" (after Platt scaling)")

    # Generate updated PD distribution plot
    plot_pd_distribution(y_prob_cal, y_test, suffix="_calibrated")

    # ── Generate calibrated PDs for full dataset ─────────────────
    print("\n  Generating calibrated PD estimates for full dataset ...")
    print("  Reading model_dataset.parquet ...")
    full = read_parquet_safe(DATA_PROCESSED / "model_dataset.parquet")
    full = full.dropna(subset=FEATURES)

    print("  Applying calibrated model ...")
    full["pd_logistic"] = calibrated_predict_proba(full[FEATURES])

    print(f"  Calibrated PD statistics:")
    print(f"    Mean PD : {full['pd_logistic'].mean()*100:.3f}%")
    print(f"    Min PD  : {full['pd_logistic'].min()*100:.4f}%")
    print(f"    Max PD  : {full['pd_logistic'].max()*100:.3f}%")
    print(f"    % below Basel floor (0.03%): "
          f"{(full['pd_logistic'] < 0.0003).mean()*100:.2f}%")

    # Save using safe write (temp file + rename)
    print("  Saving model_dataset.parquet ...")
    write_parquet_safe(full, DATA_PROCESSED / "model_dataset.parquet")
    print(f"  Saved calibrated pd_logistic to model_dataset.parquet")

    # Save both pipeline and calibration layer together
    joblib.dump(
        {"pipeline": model_pipeline, "calibration_lr": calibration_lr},
        DATA_PROCESSED / "logistic_pipeline_calibrated.joblib"
    )
    print(f"  Saved logistic_pipeline_calibrated.joblib")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Week 7: Logistic regression PD model")
    print("=" * 60)

    print("\n[1/6] Loading data ...")
    train, test = load_data()

    print("\n[2/6] Training model ...")
    model_pipeline = build_and_train(train)

    print("\n[3/6] Evaluating on test set ...")
    metrics, y_prob = evaluate(model_pipeline, test)

    print("\n[4/6] Inspecting coefficients ...")
    print_coefficients(model_pipeline)

    print("\n[5/6] Generating plots ...")
    y_test = test[TARGET].values
    plot_roc_curve(y_test, y_prob, metrics["auc"])
    plot_calibration(y_test, y_prob)
    plot_pd_distribution(y_prob, y_test)

    print("\n[6/6] Calibrating and generating PD estimates ...")
    calibrate_and_generate_pd(model_pipeline, train, test)

    # Save the base (uncalibrated) pipeline too — useful for reference
    joblib.dump(model_pipeline,
                DATA_PROCESSED / "logistic_pipeline.joblib")
    print(f"\n  Base pipeline saved to logistic_pipeline.joblib")

    print("\nWeek 7 complete. Ready for week 8 — XGBoost.")


if __name__ == "__main__":
    main()