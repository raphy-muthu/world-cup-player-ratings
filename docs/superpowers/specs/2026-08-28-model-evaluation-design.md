# Model evaluation — design

Covers roadmap Steps 4-9. This session stops at Step 7 (baseline), so Steps 7-9
are specified here but implemented later.

## Decisions

**Drop `nationality_code`.** One-hot encoding it would turn 1 column into 41 on
only 305 rows (4 countries have a single player each), risking overfitting and
diluting regularization pressure on genuinely predictive features. Same
reasoning that already excluded `Squad`. Feature count drops 18 -> 17.

**One-hot encoding lives in the modeling script, not `prepare_model_data.py`.**
`model_features.csv` stays human-readable (`DEF`/`MID`/`FWD` as text) so it can
be eyeballed and reused. Converting text to numbers is a requirement of
scikit-learn/XGBoost specifically, so it belongs next to the code that needs it.

**Three model candidates, not five.** `Ridge` represents the linear family
(skipping Lasso/ElasticNet keeps the comparison to three real contenders),
plus `RandomForestRegressor` and `XGBRegressor`. No model is pre-selected —
the comparison decides, with interpretability weighed against any accuracy
gain, since explaining the methodology is a stated project priority.

## Files

**`src/explore_features.py` (Step 4).** Exploratory pass over
`model_features.csv`: null check, distribution summary, target distribution,
near-zero-variance check, and a correlation matrix flagging feature pairs above
|r| = 0.8. Findings are reported, not silently acted on — a strong correlation
is a decision to surface, not something to auto-drop.

**`src/evaluate_models.py` (Steps 5, 6, then later 7-9).** One-hot encodes
`position` (3 categories, `drop_first=True` -> 2 columns), builds a 5-fold
cross-validation splitter stratified by position so DEF/MID/FWD stay balanced
across folds, with a fixed seed for reproducibility. Later: baseline
(Step 7), the three candidates (Step 8), and CV evaluation with MAE/R²
(Step 9), saving results to `data/processed/model_comparison_results.csv`.

## Verification

- `prepare_model_data.py` still passes all assertions after the feature drop;
  output is 305 rows with 17 features and no `nationality_code`
- Encoded matrix is fully numeric with no nulls
- Each CV fold's position distribution is checked against the overall
  distribution to confirm stratification actually worked
- Every fold's train/test indices are disjoint and together cover all 305 rows
