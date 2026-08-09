"""
DOCSTRING
"""

import pandas as pd

from utils import normalizeName

fbref = pd.read_csv("data/raw/players_data_light-2025_2026.csv")

print(fbref.head())


def nationClean(nationValue):
    if pd.isna(nationValue):
        return None

    clean_nation = nationValue.split()
    if len(clean_nation) != 2:
        return None
    else:
        return(clean_nation[1])
    


fbref["nationality_code"] = fbref["Nation"].apply(nationClean)

print(fbref["nationality_code"].isna().sum())

# 3 players have missing nations (Nathan Mbala, Luis Orejuela, Yael Trepy). Can get nation from google
print(fbref[fbref['nationality_code'].isna()][['Player', 'Nation', 'Squad', "Pos"]])

print("\nSaved to data/processed/fbref_cleaned.csv")

fbrefGrouped = fbref.groupby("Player").size()

duplicateNames = fbrefGrouped[fbrefGrouped > 1]
print(f"Duplicate Names: {duplicateNames}")

fbrefFilter = fbref[fbref["Player"].isin(duplicateNames.index)]

# WHEN DOING FUZZY MATCHING: Must also match players from duplicateNames with the club to avoid statistical overlap
print(fbrefFilter.sort_values("Player"))

fbref = fbref.drop(columns=['PK_stats_shooting', 'PKatt_stats_shooting', 'CrdY_stats_misc', 'CrdR_stats_misc'])
count_columns = [
    'MP', 'Starts', 'Min', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt',
    'CrdY', 'CrdR', 'G+A-PK', 'Sh', 'SoT', 'Crs', 'TklW', 'Int', 'Fld',
    'Fls', 'OG', 'GA', 'SoTA', 'Saves', 'W', 'D', 'L', 'CS',
    'PKatt_stats_keeper', 'PKA', 'PKsv', 'PKm'
]

# Handle mid-season transfers: combine split rows into one per player
# 151 players have two rows in FBref because of a mid-season club transfer

# Columns that represent counts and should be SUMMED across a player's rows
count_columns = [
    'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR',
    'TklW', 'Int', 'Crs', 'Fls'
]

agg_rules = {col: 'sum' for col in count_columns}
agg_rules['Squad'] = lambda squads: ' / '.join(squads.unique())
agg_rules['nationality_code'] = 'first'
agg_rules['Nation'] = 'first'
# add 'first' for any other non-numeric column you want to keep (e.g., Pos, Age)

fbref_aggregated = fbref.groupby('Player', as_index=False).agg(agg_rules)

print(f"Rows before aggregation: {len(fbref)}")
print(f"Rows after aggregation: {len(fbref_aggregated)}")
print(f"Unique player names: {fbref['Player'].nunique()}")
assert len(fbref_aggregated) == fbref['Player'].nunique(), "Aggregation didn't collapse duplicates correctly!"

# Spot check a known transferred player to confirm it worked as expected
print(fbref_aggregated[fbref_aggregated['Player'] == 'Adama Traoré'])

fbref_aggregated['name_norm'] = fbref_aggregated['Player'].apply(normalizeName)

fbref_aggregated.to_csv("data/processed/fbref_cleaned.csv", index=False)
print("Saved to data/processed/fbref_cleaned.csv")

assert len(fbref_aggregated) == fbref_aggregated['Player'].nunique(), \
    "Duplicate player rows still exist after aggregation!"

assert 'name_norm' in fbref_aggregated.columns, "name_norm column is missing!"
assert fbref_aggregated['name_norm'].isna().sum() == 0, \
    "Some name_norm values are missing!"

missing_nat = fbref_aggregated['nationality_code'].isna().sum()
assert missing_nat <= 3, \
    f"Expected at most 3 missing nationality codes, found {missing_nat}"

for col in ['PK_stats_shooting', 'PKatt_stats_shooting', 'CrdY_stats_misc', 'CrdR_stats_misc']:
    assert col not in fbref_aggregated.columns, f"{col} should have been dropped!"

# 5. Known transferred player should show combined minutes, not a partial season
traore_row = fbref_aggregated[fbref_aggregated['Player'] == 'Adama Traoré']
assert len(traore_row) == 1, "Adama Traoré should appear exactly once after aggregation"
assert traore_row['Min'].values[0] > 300, "Adama Traoré's minutes look too low — aggregation may have failed"

# 6. Sanity check on row count — should be well under the pre-aggregation total
assert len(fbref_aggregated) < 2839, "Row count didn't shrink — aggregation may not have run"

print("All load_fbref.py validation checks passed.")