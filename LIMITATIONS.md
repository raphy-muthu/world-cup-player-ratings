# Known limitations — wc_fbref_final.csv

Running log of data-quality and methodology limitations found while building the
WC↔FBref join and rating model. Add to this as new issues surface.

## Coverage

- **FBref is Big-5-leagues-only.** `fbref_cleaned.csv` covers Premier League, La Liga,
  Bundesliga, Serie A, Ligue 1 (96 squads total). 369 of 707 WC players (52%) play
  outside this coverage (MLS, Liga MX, Saudi Pro League, Süper Lig, South African
  PSL, Portuguese Primeira Liga, etc.) and can never be matched — this is a hard
  ceiling on match rate, not a matching-quality problem. See
  `data/processed/wc_still_unmatched.csv`.

## Name-matching

- **Fuzzy matching is nationality-restricted and threshold-gated (token_set_ratio
  >= 85), with club/squad agreement as a corroborating signal** — see
  `src/fuzzy_match.py`. 18 borderline cases required manual review
  (`manual_verdict` column in `data/processed/fuzzy_match_review.csv`); 7 were
  confirmed false positives (common-name collisions, e.g. two different players
  both named "Pedro") and excluded.
- **Accent-stripping (`utils.py:normalizeName`) silently drops characters that
  don't decompose under Unicode NFKD**, instead of transliterating them —
  e.g. "Sørloth" → "Srloth", not "Sorloth". Fuzzy matching recovers most of these,
  but it's a latent bug in the normalization step itself, not just a downstream
  symptom.

## Upstream data quality (wc_merged.csv)

- **Glued-name bug**: 6 of 707 players have their surname concatenated onto the
  front of the name field with no space (e.g. `"MARQUINHOSMarcos"`,
  `"MARTINELLIGabriel Gabriel"`). Confirmed via full-dataset regex scan
  (`[A-Z]{2,}[a-z]` pattern in `player_name`). Of the 6:
  - 1 (Guimarães) was recovered by fuzzy matching anyway (lucky — token
    resorting happened to still line up)
  - 2 (Marquinhos/PSG, Martinelli/Arsenal) play for Big-5 clubs and are
    presumably in FBref, but the corrupted tokens broke even
    `token_set_ratio` — these are real, recoverable losses, not coverage gaps
  - 3 are moot (their clubs aren't in FBref's coverage regardless)
  - Fix belongs upstream, in whatever step produces `wc_merged.csv`
    (`src/join_data.py` or its raw source), not in the matching step.
- **Truncated names**: at least 2 players in the still-unmatched set have a
  `player_name` of literally `"Mc"` — clearly cut off during upstream parsing
  (likely something like "McGinn"). Same fix location as above.

## FBref data quality (carried over from `load_fbref.py`)

- **3 players have no `nationality_code`**: Luis Orejuela (Mallorca), Nathan Mbala
  (Metz), Yael Trepy (Cagliari). Their `Nation` field didn't parse into exactly
  two tokens during `load_fbref.py`'s `nationClean()` step. Flagged there
  already but not fixed — these 3 fall back to the full (non-nationality-restricted)
  candidate pool during fuzzy matching, which is a slightly weaker match
  guarantee for whichever WC players might have matched to them.

## Modeling implications

- **GK sample size (n=15 in the matched set) is too small for a standalone
  position-split model.** DEF/MID/FWD (n=88–111 each) are borderline —
  workable but thin once more than a few features are added.
  Recommendation: single combined model with position as a categorical
  feature, not four separate models. See `notebooks/` (if position-split vs.
  combined-model decision is written up there — otherwise this doc is the
  record of that decision).

- **RESOLVED — `relative_score` now measures the project's core premise.**
  It previously z-scored WC rate and club rate *independently* and averaged
  them, which measures general quality across both contexts rather than a
  within-player delta. Caught because Michael Olise, Ousmane Dembélé, and
  Alphonso Davies all *underperformed their own club rate* at the WC yet
  ranked in the top 10. Fixed by computing the per-player delta first and
  standardizing that: `z(WC_rate - club_rate)`. After the fix those three fall
  to ranks 192, 234, and 301 of 320 — correctly identified as underperforming
  their own baseline rather than rewarded for a high baseline.

- **Discipline is an absolute deduction, not a relative one.** Yellows, reds,
  and own goals subtract flat amounts (-1.00 / -2.50 / -1.75) rather than
  being z-scored within position. The z-scored version distorted badly: since
  most players at a position have zero cards, the group's spread was tiny and
  any nonzero value looked extreme against it — 2 yellows in 95 minutes once
  produced a -6.3 penalty, roughly five standard deviations of the production
  score. A failed intermediate attempt to fix this by lowering the *weights*
  made it worse (it tightened the spread further, pushing that player's
  z-score to -7.9), which is what motivated abandoning z-scoring for cards
  entirely. Known data limitation: `yellow_cards` and `red_cards` are separate
  counts, so a red earned via a second yellow is indistinguishable from a
  straight red; every red is charged the direct-red rate. Affects 5 of 320
  players who carry both yellows and a red.

- **Feature leakage for Phase 4 (predictive modeling).** `final_rating` is
  built directly from: `yellow_cards`, `red_cards`, `own_goals`, WC
  `minutes_played`, `Gls`, `Ast`, club `Min` (FBref), `saves`, `clean_sheets`,
  `goals_conceded`, `matches_played`. None of these 11 columns may be used as
  a predictive feature — doing so would let a model partially reconstruct its
  own label instead of predicting it, inflating validation metrics in a way
  that wouldn't hold up under scrutiny. (`own_goals` was a feature until the
  target switched to absolute deductions; it moved onto this list at that
  point.) Note that club-side `CrdY`/`CrdR` are *not* leakage — the target
  uses World Cup cards, these are club-season cards, different data.

- **`match_type` must also be excluded from any feature set**, for a different
  reason than the 11 above: it's not part of a player's season, it's an
  artifact of this project's own name-matching pipeline (exact vs. fuzzy
  join). Including it risks the model picking up a spurious correlation
  between match confidence and rating that has nothing to do with football.

- **`shots` and `shots_on_target` are 100% null in the raw source data**
  (`data/raw/world-cup-2026/player_stats.csv`, confirmed across all 1248 rows
  before any filtering — every other stat column in that file, e.g. `assists`,
  is fully populated). Traced through every pipeline stage
  (`player_stats.csv` -> `wc_merged.csv` -> `wc_fbref_final.csv`) and the
  emptiness is present from the very first byte — this is a genuine gap in
  the upstream data source, not a bug introduced by any join/merge in this
  project. Excluded from the Phase 4 feature set for this reason, unrelated
  to the leakage issue above — there's no data here to leak or use.

- **Locked feature set for Phase 4 (15 features, 305 outfielders):**
  `position`, `caps`, `career_goals`, `matches_started`, `penalty_goals`,
  `market_value_eur`, `height_cm`, `age`, `Starts`, and club-side
  `TklW`/`Int`/`Crs`/`Fls`/`CrdY`/`CrdR` as per-match rates. Zero nulls.
  Four candidates were considered and excluded:
  - `Squad` — high-cardinality (many distinct clubs) relative to n=305
  - `nationality_code` — 41 distinct values on 305 rows (4 countries have a
    single player), so one-hot encoding would add 41 sparse columns; same
    overfitting logic as `Squad`
  - `MP` — correlated r=0.865 with `Starts` (near-duplicate). Keeping `Starts`
    instead gave a lower CV MAE in 20/20 random seeds. `MP` is still used
    internally as the per-match rate denominator, just not as a feature.
  - `own_goals` — moved to the leakage list when it entered the target

- **Near-zero-variance features retained:** `CrdR_per_match` (var 0.0003) and
  `CrdY_per_match` (0.0088). Red cards are rare enough that the column is
  nearly constant. Left in rather than dropped — regularization can handle
  them — but they carry little usable signal.

- **RESOLVED (Steps 8-12) — `final_rating` (or `relative_score`) as an exact
  regression target is not predictable from these features, but a coarser
  version of the same question is.** This was re-checked properly after the
  absolute-discipline-penalty change, not left as the stale pre-change
  estimate. Three regressors (RidgeCV, RandomForest, XGBoost) under 5-fold
  stratified CV vs. a predict-the-mean/median baseline (MAE 0.7712):
  RidgeCV MAE 0.7659/R²-0.023, RandomForest MAE 0.7589/R²-0.001, XGBoost
  MAE 0.7604/R²+0.004 — technically the only one clearing R²>0, but its
  predictions varied 42% as much as actual `final_rating` while explaining
  under 1% of variance, meaning it was making confident-looking guesses that
  were mostly wrong. Residual analysis (Step 11) found a large, structural
  bias: carded players' ratings were over-predicted by 0.71 on average (they
  can't be seen coming from a club profile), clean players' under-predicted by
  0.34. Re-running against `relative_score` alone (discipline removed) made
  every model *worse*, not better — ruling out "the noisy discipline term is
  masking real performance signal" as the explanation. `relative_score` is WC
  rate minus club rate, which deliberately cancels the stable, player-specific
  component the features describe; there is no reason to expect it to be
  regressable, and it isn't.

  Reframing as binary classification — did the player over- or under-perform
  their own club baseline (`relative_score > 0`, a near-balanced 157/148
  split) — DID find real signal: LogisticRegression scored AUC 0.590 (std
  0.014 across 25 CV seeds, above 0.55 in 25/25). Verified against a
  permutation test (labels shuffled 40x): null AUC 95th percentile = 0.566,
  real AUC clears it — statistically distinguishable from noise, not a lucky
  split. Driven mainly by `Crs_per_match` (-0.43, crossing-heavy players
  underperform their own baseline more) and `market_value_eur` (-0.35,
  regression to the mean — elite players have less room to overshoot their
  own high baseline). Ceiling stated plainly: AUC 0.59 is weak — a real, small
  directional tendency, not a tool for confident predictions about any one
  player.

  **Shipped:** the classifier only (`models/over_underperformance_classifier.joblib`,
  built by `src/train_final_model.py`), not the regressor — there's nothing
  in the regression results that outperforms guessing, so nothing to ship
  there. The regressor's null result is documented here, not as an artifact.
