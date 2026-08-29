"""
Step 4: Exploratory pass over model_features.csv before any modeling.
Design: docs/superpowers/specs/2026-08-28-model-evaluation-design.md

Reports findings rather than acting on them. A strong feature correlation or a
skewed distribution is a decision to surface, not something to silently drop —
same principle applied to the fuzzy-match red flags and the rating leaderboard
sanity check earlier in this project.
"""

import pandas as pd

CORR_FLAG_THRESHOLD = 0.8
NEAR_ZERO_VAR_THRESHOLD = 0.01

pd.set_option("display.width", 200)

df = pd.read_csv("data/processed/model_features.csv")
target = "final_rating"
id_cols = ["player_id", "player_name"]
feature_cols = [c for c in df.columns if c not in id_cols + [target]]
numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

print(f"Rows: {len(df)} | Features: {len(feature_cols)} "
      f"({len(numeric_features)} numeric, {len(feature_cols) - len(numeric_features)} categorical)")

# --- Nulls: re-verifying rather than assuming prepare_model_data.py's assertion holds ---
null_counts = df.isna().sum()
print(f"\nNull check: {null_counts.sum()} total nulls across all columns")
if null_counts.sum() > 0:
    print(null_counts[null_counts > 0])

# --- Distributions ---
print("\n=== Feature distributions ===")
print(df[numeric_features].describe().T[["mean", "std", "min", "50%", "max"]].round(3))

print(f"\n=== Target distribution ({target}) ===")
print(df[target].describe().round(3))

print("\n=== Categorical features ===")
for col in feature_cols:
    if col not in numeric_features:
        print(f"{col}: {dict(df[col].value_counts())}")

# --- Near-zero variance: a feature that barely varies can't help a model and can
#     destabilize scaling/regularization ---
print("\n=== Near-zero-variance check ===")
variances = df[numeric_features].var(ddof=1)
low_var = variances[variances < NEAR_ZERO_VAR_THRESHOLD]
if len(low_var) > 0:
    print(f"Features with variance < {NEAR_ZERO_VAR_THRESHOLD}:")
    print(low_var.round(6))
else:
    print(f"None below {NEAR_ZERO_VAR_THRESHOLD}")

# --- Correlation: highly correlated features are near-duplicates. Matters most for
#     linear models, where two collinear features split/destabilize a coefficient ---
print(f"\n=== Feature pairs with |correlation| > {CORR_FLAG_THRESHOLD} ===")
corr = df[numeric_features].corr()
flagged = []
for i, a in enumerate(numeric_features):
    for b in numeric_features[i + 1:]:
        r = corr.loc[a, b]
        if abs(r) > CORR_FLAG_THRESHOLD:
            flagged.append((a, b, round(r, 3)))
if flagged:
    for a, b, r in sorted(flagged, key=lambda x: -abs(x[2])):
        print(f"  {a} <-> {b}: r = {r}")
else:
    print(f"  None above {CORR_FLAG_THRESHOLD}")

# --- Feature/target correlation: which features look predictive at all, before
#     any model is fit. Weak correlations here are informative, not disqualifying —
#     a feature can matter in combination without correlating on its own. ---
print(f"\n=== Correlation of each feature with {target} ===")
target_corr = df[numeric_features].corrwith(df[target]).sort_values(key=abs, ascending=False)
print(target_corr.round(3).to_string())
