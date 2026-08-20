"""
Step 3: Combine exact name_norm matches with confirmed fuzzy matches into the
final wc <-> fbref joined table.

A fuzzy match is included only if:
  - it cleared the token_set_ratio threshold AND club/squad corroborated it
    (meets_threshold=True, red_flag=False), or
  - it was red-flagged but a human confirmed it's real (manual_verdict='accept')

Rows with manual_verdict='reject' (confirmed false positives) are excluded.
"""

import pandas as pd

wc = pd.read_csv("data/processed/wc_merged.csv")
fbref = pd.read_csv("data/processed/fbref_cleaned.csv")
review = pd.read_csv("data/processed/fuzzy_match_review.csv")

# Exact matches
exact = wc.merge(fbref, on="name_norm", how="inner")
exact["match_type"] = "exact"
exact_wc_ids = set(exact["player_id"])

# Confirmed fuzzy matches: clean accepts, plus red-flagged rows manually confirmed real
confirmed_fuzzy = review[
    (review["meets_threshold"] & ~review["red_flag"])
    | (review["manual_verdict"] == "accept")
].copy()

fuzzy = (
    wc[wc["player_id"].isin(confirmed_fuzzy["wc_player_id"])]
    .merge(
        confirmed_fuzzy[["wc_player_id", "fbref_matched_name"]],
        left_on="player_id", right_on="wc_player_id",
    )
    .merge(fbref, left_on="fbref_matched_name", right_on="Player")
    .drop(columns=["wc_player_id", "fbref_matched_name"])
)
fuzzy["match_type"] = "fuzzy"

final = pd.concat([exact, fuzzy], ignore_index=True)
assert final["player_id"].is_unique, "A WC player ended up matched more than once!"

final.to_csv("data/processed/wc_fbref_final.csv", index=False)

# --- Reporting ---
n_total = len(wc)
n_exact = len(exact)
n_fuzzy = len(fuzzy)
n_matched = n_exact + n_fuzzy
n_unmatched = n_total - n_matched

print(f"Final match rate: {n_matched}/{n_total} ({n_matched/n_total:.1%})")
print(f"  Exact matches: {n_exact}")
print(f"  Fuzzy matches (confirmed): {n_fuzzy}")
print(f"  Still unmatched: {n_unmatched}")

still_unmatched = wc[~wc["player_id"].isin(exact_wc_ids) & ~wc["player_id"].isin(fuzzy["player_id"])].copy()

from rapidfuzz import fuzz, process
fb_squads = list(set(fbref["Squad"].str.split(" / ").explode().str.strip().unique()))

def big5_plausible(club):
    _, score, _ = process.extractOne(club, fb_squads, scorer=fuzz.token_sort_ratio)
    return score

still_unmatched["club_squad_score"] = still_unmatched["club_team"].apply(big5_plausible)
outside_big5 = (still_unmatched["club_squad_score"] < 80).sum()
plausibly_big5_but_unmatched = (still_unmatched["club_squad_score"] >= 80).sum()

print(f"\nBreakdown of {n_unmatched} still-unmatched players:")
print(f"  Club not plausibly a Big-5 squad (out of FBref's coverage entirely): {outside_big5}")
print(f"  Club plausibly Big-5 but still unmatched (below fuzzy threshold / rejected as false positive): {plausibly_big5_but_unmatched}")

still_unmatched.to_csv("data/processed/wc_still_unmatched.csv", index=False)
print("\nSaved final table to data/processed/wc_fbref_final.csv")
print("Saved still-unmatched breakdown to data/processed/wc_still_unmatched.csv")
