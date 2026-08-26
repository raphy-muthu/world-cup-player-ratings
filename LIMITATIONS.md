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

- **`relative_score`/`final_rating` (in `player_ratings.csv`) do not currently
  measure the project's core premise.** The project's stated goal is to rate a
  player by deviation from their own club-season baseline — did they over- or
  under-perform their normal level at the tournament. The current formula in
  `build_ratings.py` instead z-scores WC rate and club rate *independently*
  against position peers and averages them, which measures general quality
  across both contexts, not a within-player before/after delta. Confirmed with
  real examples: Michael Olise, Ousmane Dembélé, and Alphonso Davies all
  *underperformed their own club rate* at the WC (negative WC-minus-club
  delta) yet rank in the current top 10 by `final_rating` anyway. Planned fix:
  redesign around `z(WC_rate - club_rate)` per player, not `z(WC_rate)` and
  `z(club_rate)` averaged. Not yet implemented — pending design review.

- **Feature leakage risk for Phase 4 (predictive modeling).** `final_rating`
  is built directly from: `yellow_cards`, `red_cards`, WC `minutes_played`,
  `Gls`, `Ast`, club `Min` (FBref), `saves`, `clean_sheets`, `goals_conceded`,
  `matches_played`. None of these 10 columns should be used as a predictive
  feature for `final_rating` (or any target derived from it) — doing so would
  let a model partially reconstruct its own label instead of predicting it,
  inflating validation metrics in a way that wouldn't hold up under scrutiny.
  Everything else on the original candidate feature list is clean: `position`,
  `caps`, `career_goals`, `matches_started`, `penalty_goals`, `own_goals`,
  and club-side `MP`/`Starts`/`TklW`/`Int`/`Crs`/`Fls`/`CrdY`/`CrdR`.
