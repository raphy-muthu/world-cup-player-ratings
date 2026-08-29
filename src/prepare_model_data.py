"""
Step 4 prep: build a clean, modeling-ready dataset from wc_fbref_final.csv +
player_ratings.csv. Design: docs/superpowers/specs/2026-08-28-model-data-prep-design.md

Two decisions baked in here:
- GKs are excluded entirely. final_rating means a structurally different
  thing for GKs (rate-vs-peers, no club baseline exists) than for outfielders
  (delta-vs-own-baseline) — mixing both under one label would corrupt what a
  model learns, and n=15 GKs is too small to validate a GK-specific model
  regardless. GK final_rating stays available as a descriptive number from
  build_ratings.py, just outside this model's scope.
- TklW/Int/Crs/Fls/CrdY/CrdR are season-cumulative counts that scale with
  club playing time, so they're converted to per-match rates using MP before
  being used as features — same normalization problem the target itself had
  before shrinkage, now handled on the input side. MP is the denominator
  (not Min) because Min already built final_rating and is on the leakage list.
"""

import pandas as pd

df = pd.read_csv("data/processed/wc_fbref_final.csv")
ratings = pd.read_csv("data/processed/player_ratings.csv")

EXCLUDED_LEAKAGE_COLS = [
    "yellow_cards", "red_cards", "own_goals", "minutes_played", "Gls", "Ast", "Min",
    "saves", "clean_sheets", "goals_conceded", "matches_played", "match_type",
]

outfield = df[df["position"] != "GK"].copy()
print(f"Excluded {len(df) - len(outfield)} GK rows. {len(outfield)} outfielders remain.")

assert (outfield["MP"] > 0).all(), \
    "Found club MP <= 0 in outfielders — per-match rate would divide by zero"

RATE_COLS = ["TklW", "Int", "Crs", "Fls", "CrdY", "CrdR"]
for col in RATE_COLS:
    outfield[f"{col}_per_match"] = outfield[col] / outfield["MP"]

outfield["age"] = (
    pd.Timestamp("2026-06-11") - pd.to_datetime(outfield["date_of_birth"])
).dt.days / 365.25

# nationality_code is deliberately NOT a feature. One-hot encoding it would turn
# 1 column into 41 on only 305 rows (4 countries have a single player each), which
# risks overfitting and dilutes regularization pressure on features that actually
# predict. Same reasoning that already excluded Squad.
# own_goals was a feature until the target switched to absolute discipline
# deductions — it now helps BUILD final_rating (-1.75 each), so using it to
# predict final_rating would be leakage, same rule as yellow_cards/red_cards.
feature_cols = [
    "position", "caps", "career_goals", "matches_started", "penalty_goals",
    "market_value_eur", "height_cm", "age",
    # MP dropped: it correlated r=0.865 with Starts (near-duplicate signal), and
    # keeping Starts instead produced a lower CV MAE in 20/20 random seeds.
    # MP is still used above as the per-match rate denominator.
    "Starts",
    "TklW_per_match", "Int_per_match", "Crs_per_match", "Fls_per_match",
    "CrdY_per_match", "CrdR_per_match",
]
id_cols = ["player_id", "player_name"]

model_data = outfield[id_cols + feature_cols].merge(
    ratings[["player_id", "final_rating"]], on="player_id", how="left"
)

# --- Verification, not assumption ---
assert len(model_data) == len(outfield), \
    f"Join changed row count: {len(outfield)} -> {len(model_data)}"
assert model_data["player_id"].is_unique, "A player ended up duplicated by the join"
assert not any(col in model_data.columns for col in EXCLUDED_LEAKAGE_COLS), \
    "A leakage column leaked into the model dataset"
assert model_data.notna().all().all(), \
    f"Nulls found in: {model_data.columns[model_data.isna().any()].tolist()}"

model_data.to_csv("data/processed/model_features.csv", index=False)

print(f"\nSaved {len(model_data)} rows, {len(model_data.columns)} columns to "
      f"data/processed/model_features.csv")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"Target: final_rating")
