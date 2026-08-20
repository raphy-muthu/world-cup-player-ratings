"""
Step 4: Build final_rating for each matched player from wc_fbref_final.csv.

Design:
- Outfielders (DEF/MID/FWD): blend WC-tournament and club-season goal/assist
  productivity (per-90), standardized within position, minus a card-discipline
  penalty (also standardized within position).
- GKs: fbref_cleaned.csv carries no goalkeeper stats (Saves/GA/CS were dropped
  in load_fbref.py), so club-side comparison isn't possible for them. GKs get
  their own formula built only from WC-side keeper stats (saves, goals
  conceded, clean sheets), standardized within the GK group only.

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

CARDS are deliberately left RAW/unshrunk. Yellow/red cards are meant to have
a high, direct impact on final_rating by design — shrinking them toward the
position mean would blunt that on purpose, which is the opposite of what's
wanted here even though it also reduces the influence of small-sample noise
on the card penalty.
"""

import numpy as np
import pandas as pd

K90_TRUST_UNITS = 3  # ~270 minutes of prior weight for per-90 metrics
K_MATCH_TRUST_UNITS = 3  # ~3 matches of prior weight for match-based rates (e.g. clean sheets)

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
outfield["cards_per90_raw"] = (outfield["yellow_cards"] + 2 * outfield["red_cards"]) / outfield["minutes_played"] * 90

# --- Shrunk per-90 rates for PRODUCTION metrics only: blend each player's own
#     rate with their position's aggregate rate, weighted by minutes played.
#     Cards are intentionally NOT shrunk — cards are meant to have a high,
#     direct impact on final_rating, not be smoothed toward the position mean. ---
outfield["wc_goals_per90"] = shrink_rate(outfield, "tournament_goals", "minutes_played", "position", K90_TRUST_UNITS)
outfield["wc_assists_per90"] = shrink_rate(outfield, "assists", "minutes_played", "position", K90_TRUST_UNITS)
outfield["club_goals_per90"] = shrink_rate(outfield, "Gls", "Min", "position", K90_TRUST_UNITS)
outfield["club_assists_per90"] = shrink_rate(outfield, "Ast", "Min", "position", K90_TRUST_UNITS)
outfield["cards_per90"] = outfield["cards_per90_raw"]

# --- Position-level standard deviations (reported explicitly, not just consumed) ---
production_metrics = ["wc_goals_per90", "wc_assists_per90", "club_goals_per90", "club_assists_per90"]
print("\nPosition-level std (outfielders, SHRUNK production rates; cards is RAW, unshrunk):")
print(outfield.groupby("position")[production_metrics + ["cards_per90"]].std(ddof=1).round(3))
print("Position-level std (outfielders, RAW rates, for comparison):")
print(outfield.groupby("position")[[c + "_raw" for c in production_metrics] + ["cards_per90_raw"]].std(ddof=1).round(3))

# --- Z-scores within position, average into relative_score ---
outfield, z_cols = zscore_within_group(outfield, production_metrics, "position")
outfield["relative_score"] = outfield[z_cols].mean(axis=1)

# --- Card penalty: standardized within position, subtracted ---
outfield, card_z_cols = zscore_within_group(outfield, ["cards_per90"], "position")
outfield["card_penalty"] = outfield["z_cards_per90"]
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
# in a short match sample." Cards are the exception, same as for outfielders:
# left raw/unshrunk so they keep their full, direct impact on final_rating.
gk["position"] = "GK"  # single group, but keep using the same groupby machinery

gk["saves_per90"] = shrink_rate(gk, "saves", "minutes_played", "position", K90_TRUST_UNITS)
gk["goals_conceded_per90"] = shrink_rate(gk, "goals_conceded", "minutes_played", "position", K90_TRUST_UNITS)
gk["clean_sheet_rate"] = shrink_rate(gk, "clean_sheets", "matches_played", "position", K_MATCH_TRUST_UNITS, per=1)
gk["cards_per90"] = (gk["yellow_cards"] + 2 * gk["red_cards"]) / gk["minutes_played"] * 90

gk_metrics = ["saves_per90", "goals_conceded_per90", "clean_sheet_rate"]
print("\nPosition-level std (GK, SHRUNK rates; cards is RAW, unshrunk):")
print(gk[gk_metrics + ["cards_per90"]].std(ddof=1).round(3))

gk, gk_z_cols = zscore_within_group(gk, gk_metrics, "position")

# Fewer goals conceded is better, so its z-score is flipped before averaging in —
# without this, a keeper who concedes more would incorrectly score higher.
gk["relative_score"] = gk[["z_saves_per90", "z_clean_sheet_rate"]].mean(axis=1) * (2 / 3) \
    - gk["z_goals_conceded_per90"] * (1 / 3)

gk, gk_card_z_cols = zscore_within_group(gk, ["cards_per90"], "position")
gk["card_penalty"] = gk["z_cards_per90"]
gk["final_rating"] = gk["relative_score"] - gk["card_penalty"]
gk["rating_method"] = "goalkeeper"

# ============================================================
# COMBINE + SAVE
# ============================================================
keep_cols = [
    "player_id", "player_name", "position", "club_team", "nationality_code",
    "minutes_played", "relative_score", "card_penalty", "final_rating", "rating_method",
]
ratings = pd.concat([outfield[keep_cols], gk[keep_cols]], ignore_index=True)

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
