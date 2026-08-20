"""
Step 2: Fuzzy-match WC players that failed the exact name_norm join against
FBref, restricted to same-nationality candidates where possible.

This does NOT produce a final merged table. It only produces a review CSV
(data/processed/fuzzy_match_review.csv) for manual confirmation of which
fuzzy matches are correct before they get folded into the final join.

Matching is decided on token_set_ratio (tolerant of WC's extra middle-name
tokens that FBref drops). Club/squad agreement is used as an independent
signal to red-flag matches that clear the score threshold on name text alone
but don't have corroborating club evidence — e.g. a short name like "Henrique"
getting swallowed whole by an unrelated "Luis Henrique".
"""

import pandas as pd
from rapidfuzz import fuzz, process

THRESHOLD = 85
CLUB_AGREEMENT_FLOOR = 60  # below this, club/squad text doesn't corroborate the name match

wc = pd.read_csv("data/processed/wc_merged.csv")
fbref = pd.read_csv("data/processed/fbref_cleaned.csv")
teams = pd.read_csv("data/raw/world-cup-2026/teams.csv")

# wc has no nationality_code of its own — derive it from the player's national
# team (team_id -> fifa_code), which lines up with fbref's nationality_code vocabulary.
wc = wc.merge(teams[["team_id", "fifa_code"]], on="team_id", how="left")
wc = wc.rename(columns={"fifa_code": "nationality_code"})

# Exact matches (the existing left-join) — these are kept as-is, untouched by fuzzy matching.
exact = wc.merge(fbref, on="name_norm", how="inner")
exact_wc_ids = set(exact["player_id"])
unmatched = wc[~wc["player_id"].isin(exact_wc_ids)].copy()

print(f"Exact matches: {len(exact)}")
print(f"Unmatched WC players to fuzzy-match: {len(unmatched)}")

# Candidate pool: fbref players not already claimed by an exact match, so we don't
# offer an already-used fbref row as a fuzzy candidate for a second WC player.
matched_fbref_names = set(exact["name_norm"])
fbref_pool = fbref[~fbref["name_norm"].isin(matched_fbref_names)].copy()

results = []
for _, row in unmatched.iterrows():
    same_nat_pool = fbref_pool[fbref_pool["nationality_code"] == row["nationality_code"]]
    pool = same_nat_pool if len(same_nat_pool) > 0 else fbref_pool
    restricted_to_nationality = len(same_nat_pool) > 0

    if len(pool) == 0:
        results.append({
            "wc_player_id": row["player_id"],
            "wc_name": row["player_name"],
            "wc_club_team": row["club_team"],
            "wc_nationality_code": row["nationality_code"],
            "fbref_matched_name": None,
            "fbref_squad": None,
            "fbref_nationality_code": None,
            "similarity_score": 0,
            "token_set_ratio": 0,
            "club_squad_score": 0,
            "restricted_to_nationality": restricted_to_nationality,
            "meets_threshold": False,
            "red_flag": False,
        })
        continue

    # Candidate selection now uses token_set_ratio, since that's the metric the
    # threshold is judged on — picking the "best" candidate by a different metric
    # than the one used to accept/reject it would be inconsistent.
    match_norm, score, idx = process.extractOne(
        row["name_norm"], pool["name_norm"], scorer=fuzz.token_set_ratio
    )
    match_row = pool.loc[idx]
    sort_score = fuzz.token_sort_ratio(row["name_norm"], match_norm)

    # Club/squad agreement as an independent corroborating signal. FBref squads can
    # be "/"-joined (mid-season transfers), so compare against the best-matching part.
    squad_parts = [s.strip() for s in str(match_row["Squad"]).split("/")]
    club_squad_score = max(
        fuzz.token_set_ratio(str(row["club_team"]), part) for part in squad_parts
    )

    meets_threshold = score >= THRESHOLD
    # Red flag: the name text says "match" but there's no club evidence to back it up —
    # this is the signature of a short name (e.g. "Henrique") getting fully swallowed by
    # an unrelated longer name (e.g. "Luis Henrique") rather than a genuine middle-name case.
    red_flag = meets_threshold and club_squad_score < CLUB_AGREEMENT_FLOOR

    results.append({
        "wc_player_id": row["player_id"],
        "wc_name": row["player_name"],
        "wc_club_team": row["club_team"],
        "wc_nationality_code": row["nationality_code"],
        "fbref_matched_name": match_row["Player"],
        "fbref_squad": match_row["Squad"],
        "fbref_nationality_code": match_row["nationality_code"],
        "similarity_score": round(sort_score, 1),
        "token_set_ratio": round(score, 1),
        "club_squad_score": round(club_squad_score, 1),
        "restricted_to_nationality": restricted_to_nationality,
        "meets_threshold": meets_threshold,
        "red_flag": red_flag,
    })

review = pd.DataFrame(results).sort_values("token_set_ratio", ascending=False)
review.to_csv("data/processed/fuzzy_match_review.csv", index=False)

accepted = review[review["meets_threshold"]]
flagged = review[review["red_flag"]]

print(f"\nWrote {len(review)} candidate fuzzy matches to data/processed/fuzzy_match_review.csv")
print(f"  token_set_ratio >= {THRESHOLD}: {len(accepted)}")
print(f"    of which red-flagged (score says match, club/squad doesn't corroborate): {len(flagged)}")
print(f"    clean accepts (score + club agree): {len(accepted) - len(flagged)}")
print(f"  token_set_ratio < {THRESHOLD} (likely not a real match / genuinely not in FBref): "
      f"{len(review) - len(accepted)}")
print("\nNo rows have been merged into a final table. Review the CSV — pay extra attention to "
      "red-flagged rows — then re-run the finalization step with your confirmed list.")
