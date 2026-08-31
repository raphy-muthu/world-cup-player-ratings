"""
Step 12: retrain the chosen model on the full dataset and save it.

Decision trail (see LIMITATIONS.md for full detail): a regressor predicting
the exact final_rating value never beat a constant-prediction baseline
(best R2 = +0.004, not meaningfully different from 0 - see
model_comparison_results.csv). Reframing as a binary classification target -
did the player over- or under-perform their own club baseline
(relative_score > 0) - DID find real, verified signal: LogisticRegression
scored AUC 0.590 +/- 0.014 across 25 CV seeds, and a permutation test (labels
shuffled, null 95th percentile = 0.566) confirmed this is distinguishable
from noise, not a lucky split.

This ships that classifier, not the regressor. The regressor's null result is
documented but not shipped as an artifact - there's nothing to ship that
outperforms guessing.

Ceiling, stated plainly: AUC 0.59 is weak. It's a real, small, directionally
consistent tendency (crossing-heavy, high-value players tend to underperform
their own baseline; defensive-work-rate players tend to overperform it) - not
a tool for confident predictions about any individual player.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

df = pd.read_csv("data/processed/model_features.csv")
ratings = pd.read_csv("data/processed/player_ratings.csv")

m = df.merge(ratings[["player_id", "relative_score"]], on="player_id", how="left")
assert m["relative_score"].notna().all(), "Some players missing relative_score after join"

X = pd.get_dummies(
    df.drop(columns=["player_id", "player_name", "final_rating"]),
    columns=["position"], drop_first=True, dtype=float,
)
y = (m["relative_score"] > 0).astype(int)

assert X.notna().all().all(), "Nulls in the final feature matrix"
assert len(X) == len(y) == 305, f"Expected 305 outfielders, got {len(X)}"
print(f"Training on the full dataset: {len(X)} rows, {X.shape[1]} encoded features")
print(f"Class balance: {y.mean():.1%} over-performed their own baseline")

# Same hyperparameters validated under CV (Steps 9-10) - C=0.1 was not
# re-tuned here. Retraining on the full 305 rows (rather than just the CV
# folds) is standard practice once a config is validated: CV's job was
# choosing/validating the approach, not holding back data from the final
# artifact.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_SEED),
)
model.fit(X, y)

joblib.dump({
    "model": model,
    "feature_columns": list(X.columns),
    "target_definition": "relative_score > 0 (over-performed own club baseline at the WC)",
    "cv_auc_mean": 0.5902,
    "cv_auc_std": 0.0141,
    "permutation_null_95th_pct": 0.5655,
}, "models/over_underperformance_classifier.joblib")

coef = pd.Series(model.named_steps["logisticregression"].coef_[0], index=X.columns)
coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

print("\nSaved to models/over_underperformance_classifier.joblib")
print("\nFinal coefficients (standardized; + = more likely to over-perform own baseline):")
print(coef.round(4).to_string())

print("\n--- Sanity check: reload and confirm predictions match ---")
reloaded = joblib.load("models/over_underperformance_classifier.joblib")
pred_a = model.predict_proba(X)[:, 1]
pred_b = reloaded["model"].predict_proba(X)[:, 1]
assert np.allclose(pred_a, pred_b), "Reloaded model gives different predictions!"
print("Reloaded model's predictions match the in-memory model exactly.")
