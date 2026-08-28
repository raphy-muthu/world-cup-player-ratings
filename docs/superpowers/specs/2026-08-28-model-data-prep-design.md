# Model data preparation — design

## Purpose

Resolve Steps 2 and 3 of the pre-modeling roadmap (see `LIMITATIONS.md`) by
producing a single, clean, modeling-ready dataset — no model is trained yet.

## Decisions

**Step 2 — GK exclusion.** `final_rating` measures a structurally different
concept for GKs (rate-vs-peers, no club baseline available) than for
outfielders (delta-vs-own-baseline). GKs are excluded from the predictive
model entirely rather than included with a caveat or modeled separately —
n=15 is too small to validate a GK-specific model regardless, and mixing the
two concepts under one label would corrupt what the model learns for
everyone. GK `final_rating` values remain available as a descriptive number
from `build_ratings.py`, just outside this model's scope.

**Step 3 — club-side count normalization.** `TklW`, `Int`, `Crs`, `Fls`,
`CrdY`, `CrdR` are season-cumulative counts that scale with club playing
time. Converted to per-match rates using `MP` (matches played) as the
denominator — not `Min`, since `Min` is already on the leakage-exclusion
list (used to build `club_goals_per90`/`club_assists_per90`, which fed
`final_rating`). Raw counts are dropped in favor of the rate versions.

## Implementation

New script: `src/prepare_model_data.py`, matching this project's existing
one-file-per-pipeline-stage pattern.

**Inputs:** `data/processed/wc_fbref_final.csv` (features),
`data/processed/player_ratings.csv` (target — `final_rating` lives here,
not in the features file).

**Logic:**
1. Filter to `position != 'GK'`.
2. Assert `(MP > 0).all()` before dividing (guard, not assumption).
3. Compute `<col>_per_match = col / MP` for the six flagged columns.
4. Derive `age` from `date_of_birth`.
5. Join to `player_ratings.csv` on `player_id` to attach `final_rating`.
6. Assert none of the 11 excluded columns (10 leakage columns + `match_type`)
   are present in the output.
7. Assert no nulls anywhere in the output.
8. Assert row count is unchanged by the join (no player dropped/duplicated).

**Output:** `data/processed/model_features.csv`, 305 rows —
`player_id`, `player_name` (identifiers) + `position`, `caps`,
`career_goals`, `matches_started`, `penalty_goals`, `own_goals`,
`market_value_eur`, `height_cm`, `age`, `nationality_code`, `MP`, `Starts`,
`TklW_per_match`, `Int_per_match`, `Crs_per_match`, `Fls_per_match`,
`CrdY_per_match`, `CrdR_per_match` (18 features) + `final_rating` (target).

## Verification

- Row count = 305, exact column list matches the schema above
- Zero nulls in the output
- None of the 11 excluded columns present
- One player's `TklW_per_match` hand-recomputed directly from
  `wc_fbref_final.csv` to confirm the division is correct
