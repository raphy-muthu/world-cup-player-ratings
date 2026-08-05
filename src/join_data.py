"""
Step 1: Merge World Cup squad/player metadata with player match stats.
Join key: player_id (exact match, both files share this key).
"""

import pandas as pd

# Load World Cup files
squads = pd.read_csv('data/raw/world-cup-2026/squads_and_players.csv')
stats = pd.read_csv('data/raw/world-cup-2026/player_stats.csv')

print(f"Squads shape: {squads.shape}")
print(f"Stats shape: {stats.shape}")

# Merge on player_id (exact join, shared key)
wc = squads.merge(stats, on='player_id', suffixes=('', '_stats'))

# Drop the excluded rating column (third-party subjective rating, same trap
# as EA's overall_rating) and irrelevant provenance metadata
wc = wc.drop(columns=['average_rating', 'data_source', 'last_verified'])

# Validate that suffixed duplicate columns actually match before dropping
assert (wc['player_name'] == wc['player_name_stats']).all(), "player_name mismatch between files!"
assert (wc['team_id'] == wc['team_id_stats']).all(), "team_id mismatch between files!"
assert (wc['position'] == wc['position_stats']).all(), "position mismatch between files!"


# goals and goals_stats are NOT duplicates — they're different metrics:
# 'goals' = career international goals (from squads_and_players.csv)
# 'goals_stats' = tournament goals (from player_stats.csv)
# Rename for clarity instead of asserting equality
wc = wc.rename(columns={'goals': 'career_goals', 'goals_stats': 'tournament_goals'})

wc = wc.drop(columns=['player_name_stats', 'team_id_stats', 'position_stats'])

# Check GK-only stats are null for every non-GK position, non-null for GK
print("\nGK-only stat coverage by position:")
print(wc.groupby('position')[['clean_sheets', 'saves', 'goals_conceded']].apply(lambda x: x.notna().sum()))

print("Minutes Played: ")
print(wc["minutes_played"].describe())

bins = [-1, 0, 90, 270, wc["minutes_played"].max()]
labels = ["0", "1-90", "91-270", "271+"]

minuteBuckets = pd.cut(wc["minutes_played"], bins=bins, labels=labels).value_counts().sort_index()
print(minuteBuckets)

print(f"Sum of buckets: {minuteBuckets.sum()}")
print(f"Total players: {len(wc)}")


#Excluding players with under 90 minutes played (1 full match) since there needs to be at least 1 full match played to have any meaningful stats
print(f"WC Rows BEFORE filtering= {len(wc)}")


wc = wc[wc['minutes_played'] > 90]
print(f"WC Rows AFTER filtering = {len(wc)}")




print(f"Squads rows: {len(squads)}")
print(f"Stats rows: {len(stats)}")
print(f"Merged rows: {len(wc)}")
print()
print(wc.head())
print()
print("Columns:", list(wc.columns))

wc.to_csv("data/processed/wc_merged.csv", index=False)
print("\nSaved to data/processed/wc_merged.csv")


