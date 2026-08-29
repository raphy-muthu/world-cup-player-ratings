"""
Step 4: Build final_rating for each matched player from wc_fbref_final.csv.

Design:
- Outfielders (DEF/MID/FWD): relative_score measures WC-vs-own-club-baseline
  DEVIATION, not general quality. For each player, delta = WC_rate - club_rate
  is computed per metric (goals, assists), THEN standardized within position —
  not the other way around. Standardizing WC_rate and club_rate separately and
  averaging (an earlier version of this script) measures "how good is this
  player across both contexts," which is a different question and produces
  wrong answers for this project's actual goal: it ranked players who
  underperformed their own club rate at the WC (Olise, Dembele, Davies — all
  had negative WC-minus-club deltas) inside the top 10, because their high
  club numbers alone were enough to carry the average. Standardizing the delta
  itself fixes this — it can only reward a player for outperforming their OWN
  normal level, not for having a high normal level in general.
- Minus a card-discipline penalty (standardized within position, deliberately
  unshrunk — see below).
- GKs: fbref_cleaned.csv carries no goalkeeper stats (Saves/GA/CS were dropped
  in load_fbref.py), so there is no club-side keeper baseline to compute a
  delta against — the WC-vs-own-baseline concept this project is built around
  literally cannot be computed for GKs with the data available. GKs keep the
  old rate-based design instead: their own formula built only from WC-side
  keeper stats (saves, goals conceded, clean sheets), standardized against
  other GKs. This means GKs are rated on a different concept (absolute WC
  performance vs. peers) than outfielders (WC-vs-own-baseline) — a real,
  data-driven limitation, not an oversight. See LIMITATIONS.md.

Both branches guard against two real edge cases rather than assuming they
can't happen:
  1. zero minutes played (would make a per-90 rate divide-by-zero / inf)
  2. zero variability in a position group (would make a z-score divide-by-zero)

PRODUCTION rates (goals, assists — both WC and club side) are SHRUNK toward
the position's aggregate rate before z-scoring, not used raw. A raw per-90
rate from a handful of minutes is a noisy estimate, not a wrong one — e.g.
1 goal in 95 minutes extrapolates to an absurd "goals per 90" figure that
single-handedly distorted the top of an earlier, unshrunk version of this
leaderboard. Shrinkage blends each player's own rate with the position's
rate, weighted by how much playing time they actually have
(K90_TRUST_UNITS controls how many 90-minute "units" of prior belief a
player has to out-play before their own data dominates).

DISCIPLINE (yellows, reds, own goals) is an ABSOLUTE deduction, not a relative
one. Production is measured against a baseline — the player's own club rate,
then standardized against position peers — because "was this good?" only means
something in comparison. A booking needs no such comparison: it costs what it
costs regardless of how many other defenders were booked.

This also removes a real distortion. When the penalty was a z-score, a rare
card looked extreme purely because most players at a position have none, which
compressed the group's spread and inflated the outlier — 2 yellows in 95
minutes once produced a -6.3 penalty, five standard deviations of the
production score. Flat deductions make the cost legible and bounded:
-1.0 per yellow, -2.5 per red, -1.75 per own goal.
"""

import numpy as np
import pandas as pd

K90_TRUST_UNITS = 3  # ~270 minutes of prior weight for per-90 metrics
K_MATCH_TRUST_UNITS = 3  # ~3 matches of prior weight for match-based rates (e.g. clean sheets)

# Disciplinary deductions are ABSOLUTE, not relative. A yellow costs the same
# whether or not other players at the position were booked — unlike the
# production metrics, which are deliberately measured against the player's own
# club baseline and then standardized against peers.
#
# These are flat per-incident costs, so two yellows cost exactly 2x one yellow
# (matching the intended -1.0 / -2.0 schedule). No z-scoring: the previous
# z-scored version made a rare card look extreme purely because most players at
# a position have none, which compressed the group's spread and inflated the
# outlier's penalty (2 yellows in 95 minutes once produced a -6.3 penalty).
YELLOW_CARD_PENALTY = 1.00
RED_CARD_PENALTY = 2.50
OWN_GOAL_PENALTY = 1.75


def discipline_penalty(frame):
    """Flat, absolute disciplinary deduction — no z-scoring, no rate conversion.

    NOTE: the source data records yellow_cards and red_cards as separate counts,
    so a red earned via a second yellow is indistinguishable from a straight red.
    Every red is therefore charged the direct-red rate. 5 of 320 players carry
    both yellows and a red, so this affects only those rows.
    """
    return (
        YELLOW_CARD_PENALTY * frame["yellow_cards"]
        + RED_CARD_PENALTY * frame["red_cards"]
        + OWN_GOAL_PENALTY * frame["own_goals"]
    )

df = pd.read_csv("data/processed/wc_fbref_final.csv")


def shrink_rate(frame, count_col, exposure_col, group_col, K, per=90):
    """Empirical-Bayes-style shrinkage of a per-`per`-unit rate toward its
    group's AGGREGATE rate (sum(count)/sum(exposure), not the mean of
    individual per-player rates — averaging per-player rates would just
    re-import the same small-sample noise this is meant to fix).

    shrunk = (count + K * group_rate) / (exposure_units + K)

    As exposure_units -> infinity, shrunk -> the player's own raw rate.
    As exposure_units -> 0, shrunk -> the group's rate. K is the exposure
    (in `per`-sized units) at which the player's own data and the group
    prior are weighted equally.
    """
    group_rate = frame.groupby(group_col).apply(
        lambda g: g[count_col].sum() / g[exposure_col].sum() * per, include_groups=False
    )
    prior = frame[group_col].map(group_rate)
    exposure_units = frame[exposure_col] / per
    return (frame[count_col] + K * prior) / (exposure_units + K)


def zscore_within_group(frame, metric_cols, group_col):
    """Z-score each metric within each group, returning z_<col> columns.

    Groups where a metric has zero (or undefined) standard deviation get
    z=0 for that metric instead of NaN/inf from a division by zero — that's
    the correct behavior, since "everyone in this group is identical on this
    metric" means nobody should be pushed up or down by it.
    """
    z_cols = []
    for col in metric_cols:
        grp_mean = frame.groupby(group_col)[col].transform("mean")
        grp_std = frame.groupby(group_col)[col].transform("std", ddof=1)
        zero_var = grp_std.isna() | (grp_std == 0)

        if zero_var.any():
            affected_groups = sorted(frame.loc[zero_var, group_col].unique())
            print(f"  NOTE: zero-variability groups for '{col}': {affected_groups} "
                  f"(z set to 0 for these rows instead of dividing by zero)")

        z = (frame[col] - grp_mean) / grp_std
        z = z.where(~zero_var, 0.0)

        zcol = f"z_{col}"
        frame[zcol] = z
        z_cols.append(zcol)
    return frame, z_cols


# ============================================================
# OUTFIELDERS (DEF / MID / FWD)
# ============================================================
outfield = df[df["position"] != "GK"].copy()

# --- Zero-minutes guard, not an assumption ---
# join_data.py already filters WC minutes_played > 90, so this should never
# trip on the WC side. The club side (Min, from fbref) isn't filtered by this
# project at all, so it genuinely needs checking, not just assuming it's fine.
assert (outfield["minutes_played"] > 0).all(), \
    "Found WC minutes_played <= 0 in outfielders — per-90 rate would divide by zero"
assert (outfield["Min"] > 0).all(), \
    "Found club Min <= 0 in outfielders — per-90 rate would divide by zero"
print(f"Zero-minutes check passed for {len(outfield)} outfielders "
      f"(WC min minutes={outfield['minutes_played'].min()}, club min minutes={outfield['Min'].min()})")

# --- Raw per-90 rates (kept for comparison/inspection, not fed into z-scores directly) ---
outfield["wc_goals_per90_raw"] = outfield["tournament_goals"] / outfield["minutes_played"] * 90
outfield["wc_assists_per90_raw"] = outfield["assists"] / outfield["minutes_played"] * 90
outfield["club_goals_per90_raw"] = outfield["Gls"] / outfield["Min"] * 90
outfield["club_assists_per90_raw"] = outfield["Ast"] / outfield["Min"] * 90

# --- Shrunk per-90 rates for PRODUCTION metrics: blend each player's own rate
#     with their position's aggregate rate, weighted by minutes played.
#     Discipline is handled separately as a flat absolute deduction — it is not
#     a rate at all, so it is neither shrunk nor per-90'd. ---
outfield["wc_goals_per90"] = shrink_rate(outfield, "tournament_goals", "minutes_played", "position", K90_TRUST_UNITS)
outfield["wc_assists_per90"] = shrink_rate(outfield, "assists", "minutes_played", "position", K90_TRUST_UNITS)
outfield["club_goals_per90"] = shrink_rate(outfield, "Gls", "Min", "position", K90_TRUST_UNITS)
outfield["club_assists_per90"] = shrink_rate(outfield, "Ast", "Min", "position", K90_TRUST_UNITS)

# --- Position-level standard deviations (reported explicitly, not just consumed) ---
production_metrics = ["wc_goals_per90", "wc_assists_per90", "club_goals_per90", "club_assists_per90"]
print("\nPosition-level std (outfielders, SHRUNK production rates):")
print(outfield.groupby("position")[production_metrics].std(ddof=1).round(3))
print("Position-level std (outfielders, RAW rates, for comparison):")
print(outfield.groupby("position")[[c + "_raw" for c in production_metrics]].std(ddof=1).round(3))

# --- WC-vs-own-club-baseline delta (the core of this project's premise):
#     compute the per-player deviation FIRST, using the already-shrunk rates,
#     then standardize the deviation itself within position. This is the
#     opposite order from standardizing each rate separately and averaging —
#     that order measures general quality, this order measures over/under-
#     performance relative to the player's own established level. ---
outfield["delta_goals"] = outfield["wc_goals_per90"] - outfield["club_goals_per90"]
outfield["delta_assists"] = outfield["wc_assists_per90"] - outfield["club_assists_per90"]

delta_metrics = ["delta_goals", "delta_assists"]
print("\nPosition-level std of WC-vs-club delta (outfielders):")
print(outfield.groupby("position")[delta_metrics].std(ddof=1).round(3))

outfield, z_cols = zscore_within_group(outfield, delta_metrics, "position")
outfield["relative_score"] = outfield[z_cols].mean(axis=1)

# --- Discipline: flat absolute deduction, subtracted ---
outfield["card_penalty"] = discipline_penalty(outfield)
outfield["final_rating"] = outfield["relative_score"] - outfield["card_penalty"]
outfield["rating_method"] = "outfield"

# ============================================================
# GOALKEEPERS
# ============================================================
gk = df[df["position"] == "GK"].copy()
print(f"\nGK group size: n={len(gk)} — z-scores here are on a very small sample, treat with caution")

assert (gk["minutes_played"] > 0).all(), \
    "Found WC minutes_played <= 0 in GKs — per-90 rate would divide by zero"
assert (gk["matches_played"] > 0).all(), \
    "Found matches_played <= 0 in GKs — clean_sheet_rate would divide by zero"

# GKs only get WC-side stats: fbref_cleaned.csv has no keeper columns
# (Saves/GA/CS were dropped upstream in load_fbref.py), so there's no club-side
# keeper signal to blend in here — unlike the outfielders above.
#
# With only n=15 GKs, shrinkage matters here even more than for outfielders —
# there's very little data to distinguish "genuinely elite" from "got lucky
# in a short match sample." Discipline is the exception, same as for
# outfielders: a flat absolute deduction, not shrunk and not z-scored.
gk["position"] = "GK"  # single group, but keep using the same groupby machinery

gk["saves_per90"] = shrink_rate(gk, "saves", "minutes_played", "position", K90_TRUST_UNITS)
gk["goals_conceded_per90"] = shrink_rate(gk, "goals_conceded", "minutes_played", "position", K90_TRUST_UNITS)
gk["clean_sheet_rate"] = shrink_rate(gk, "clean_sheets", "matches_played", "position", K_MATCH_TRUST_UNITS, per=1)

gk_metrics = ["saves_per90", "goals_conceded_per90", "clean_sheet_rate"]
print("\nPosition-level std (GK, SHRUNK rates):")
print(gk[gk_metrics].std(ddof=1).round(3))

gk, gk_z_cols = zscore_within_group(gk, gk_metrics, "position")

# Fewer goals conceded is better, so its z-score is flipped before averaging in —
# without this, a keeper who concedes more would incorrectly score higher.
gk["relative_score"] = gk[["z_saves_per90", "z_clean_sheet_rate"]].mean(axis=1) * (2 / 3) \
    - gk["z_goals_conceded_per90"] * (1 / 3)

gk["card_penalty"] = discipline_penalty(gk)
gk["final_rating"] = gk["relative_score"] - gk["card_penalty"]
gk["rating_method"] = "goalkeeper"

# ============================================================
# COMBINE + SAVE
# ============================================================
shared_cols = [
    "player_id", "player_name", "position", "club_team", "nationality_code",
    "minutes_played",
]
# Every factor that actually feeds relative_score, plus the final outputs.
# Outfield and GK use different factors (see module docstring for why), so the
# two branches keep different columns here — concat aligns by name and fills
# NaN for whichever set doesn't apply to a given row, which is the correct,
# transparent way to show "this factor doesn't apply to this position."
outfield_cols = shared_cols + [
    "wc_goals_per90", "wc_assists_per90", "club_goals_per90", "club_assists_per90",
    "delta_goals", "delta_assists", "z_delta_goals", "z_delta_assists",
    "yellow_cards", "red_cards", "own_goals",
    "relative_score", "card_penalty", "final_rating", "rating_method",
]
gk_cols = shared_cols + [
    "saves_per90", "goals_conceded_per90", "clean_sheet_rate",
    "z_saves_per90", "z_goals_conceded_per90", "z_clean_sheet_rate",
    "yellow_cards", "red_cards", "own_goals",
    "relative_score", "card_penalty", "final_rating", "rating_method",
]
ratings = pd.concat([outfield[outfield_cols], gk[gk_cols]], ignore_index=True)

assert ratings["player_id"].is_unique, "A player ended up with more than one rating row"
assert ratings["final_rating"].notna().all(), "Some players have a null final_rating"
assert len(ratings) == len(df), f"Row count changed: {len(df)} -> {len(ratings)}"

ratings = ratings.sort_values("final_rating", ascending=False)
ratings.to_csv("data/processed/player_ratings.csv", index=False)

print(f"\nSaved {len(ratings)} ratings to data/processed/player_ratings.csv")
print("\nTop 10 by final_rating:")
print(ratings.head(10).to_string(index=False))
print("\nBottom 5 by final_rating:")
print(ratings.tail(5).to_string(index=False))
