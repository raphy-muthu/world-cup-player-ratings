"""
Steps 5-7: encode features, build the cross-validation strategy, and establish
the baseline every real model has to beat.
Design: docs/superpowers/specs/2026-08-28-model-evaluation-design.md

Steps 8-9 (model candidates, CV evaluation) are not implemented yet.

One-hot encoding lives here rather than in prepare_model_data.py: converting
text categories to numbers is a requirement of scikit-learn/XGBoost
specifically, not part of assembling a clean feature set, so model_features.csv
stays human-readable.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
RANDOM_SEED = 42

df = pd.read_csv("data/processed/model_features.csv")
target = "final_rating"
id_cols = ["player_id", "player_name"]

# --- Step 5: one-hot encode position ---
# drop_first=True: with 3 categories, 2 columns fully determine the third
# (both 0 => the dropped category). Keeping all 3 would make them perfectly
# collinear, which destabilizes linear-model coefficients.
X_raw = df.drop(columns=id_cols + [target])
X = pd.get_dummies(X_raw, columns=["position"], drop_first=True, dtype=float)
y = df[target]

assert X.notna().all().all(), "Nulls in the encoded feature matrix"
assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns), \
    "Non-numeric column survived encoding"
assert len(X) == len(y) == len(df), "Row count changed during encoding"

print(f"Encoded feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
print(f"Encoded columns: {list(X.columns)}")
print(f"position one-hot -> {[c for c in X.columns if c.startswith('position_')]} "
      f"(DEF is the implied baseline when both are 0)")

# --- Step 6: cross-validation strategy ---
# 5-fold rather than a single holdout: at n=305, a 20% test set is only ~61
# players, too small for one evaluation number to be trustworthy. K-fold uses
# every row for both training and evaluation across folds.
#
# Stratified by position so DEF/MID/FWD stay proportionally balanced in every
# fold — an unstratified split could hand one fold a disproportionate share of
# one position, and position is exactly what the target is standardized within.
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
folds = list(cv.split(X, df["position"]))

print(f"\n{N_SPLITS}-fold stratified CV (seed={RANDOM_SEED})")

overall = df["position"].value_counts(normalize=True).sort_index()
all_test_idx = set()
for i, (train_idx, test_idx) in enumerate(folds, 1):
    assert not set(train_idx) & set(test_idx), f"Fold {i}: train/test indices overlap"
    all_test_idx |= set(test_idx)
    dist = df.iloc[test_idx]["position"].value_counts(normalize=True).sort_index()
    dist_str = ", ".join(f"{pos}={dist.get(pos, 0):.3f}" for pos in overall.index)
    print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)} | {dist_str}")

assert all_test_idx == set(range(len(df))), \
    "Folds do not collectively cover every row exactly once"

print(f"  Overall:  " + ", ".join(f"{pos}={overall[pos]:.3f}" for pos in overall.index))
print(f"\nFold checks passed: no train/test overlap, all {len(df)} rows covered exactly once.")


# ============================================================
# Step 7: BASELINE
# ============================================================
# The number every real model must beat. Two variants, because the metric and
# the baseline have to agree on what "typical" means:
#   - mean   minimizes squared error, so it's the natural reference for R2
#            (R2 is DEFINED against the mean, so mean-baseline R2 is ~0)
#   - median minimizes absolute error, so it's the HARDER baseline for MAE
# final_rating is left-skewed (a long negative tail from discipline
# deductions), so these two differ and reporting only the mean would understate
# the bar. A model that beats mean-MAE but not median-MAE hasn't really beaten
# "predict the same number for everyone."
#
# Both are fit INSIDE each fold (on training data only), not on the full
# dataset — otherwise the baseline would peek at the held-out rows and the
# comparison would be unfair in the model's favor.
def cv_score(estimator, X, y, folds):
    """Return (mean MAE, mean R2) across folds, refitting on each fold."""
    maes, r2s = [], []
    for train_idx, test_idx in folds:
        est = estimator()
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = est.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        r2s.append(r2_score(y.iloc[test_idx], pred))
    return np.mean(maes), np.std(maes), np.mean(r2s)


print("\n" + "=" * 60)
print("STEP 7: BASELINE (predict one constant for everyone)")
print("=" * 60)

baselines = {
    "mean":   lambda: DummyRegressor(strategy="mean"),
    "median": lambda: DummyRegressor(strategy="median"),
}

results = {}
for name, factory in baselines.items():
    mae, mae_sd, r2 = cv_score(factory, X, y, folds)
    results[name] = (mae, r2)
    print(f"  {name:7s} baseline:  MAE = {mae:.4f} (+/-{mae_sd:.4f} across folds)   R2 = {r2:+.4f}")

best_mae = min(results.values(), key=lambda t: t[0])[0]
print(f"\nTarget spread for scale: final_rating std = {y.std(ddof=1):.4f}, "
      f"mean = {y.mean():.4f}, median = {y.median():.4f}")
print(f"\nBAR TO BEAT (Step 9): MAE below {best_mae:.4f}, and R2 above 0.0")
print("A model scoring worse than this is doing worse than ignoring every feature.")
