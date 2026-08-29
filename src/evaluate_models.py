"""
Steps 5-6: encode features and build the cross-validation strategy.
Design: docs/superpowers/specs/2026-08-28-model-evaluation-design.md

Steps 7-9 (baseline, model candidates, CV evaluation) are not implemented yet —
stopping here so the Step 4 exploratory findings can be reviewed first.

One-hot encoding lives here rather than in prepare_model_data.py: converting
text categories to numbers is a requirement of scikit-learn/XGBoost
specifically, not part of assembling a clean feature set, so model_features.csv
stays human-readable.
"""

import pandas as pd
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
print("\nStopping before Step 7 (baseline) — Step 4 findings need review first.")
